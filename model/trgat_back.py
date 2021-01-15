import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_query_attention import StructuredAttention
from .encoder import StackedEncoder
from .cnn import DepthwiseSeparableConv
from .model_utils import save_pickle, mask_logits, flat_list_of_lists, \
    find_max_triples, get_high_iou_sapns, expand_span
from .graph_att import GAttNet
from .position_emb import build_graph
import numpy as np
# from .attentionVLAD import attentionVLAD

'''
A simple GCN layer
'''
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        pass
    
    def forward(self, adj, features):
        #print(adj.shape, features.shape)
        out = torch.bmm(adj, features)
        out = self.linear(out)

        return out


class LinearWrapper(nn.Module):
    """1D conv layer"""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearWrapper, self).__init__()
        self.relu = relu
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.relu:
            return F.relu(self.conv(x), inplace=True)  # (N, L, D)
        else:
            return self.conv(x)  # (N, L, D)


class ConvLinear(nn.Module):
    """1D conv layer"""
    def __init__(self, in_hsz, out_hsz, kernel_size=3, layer_norm=True, dropout=0.1, relu=True):
        super(ConvLinear, self).__init__()
        layers = [nn.LayerNorm(in_hsz)] if layer_norm else []
        layers += [
            nn.Dropout(dropout),
            DepthwiseSeparableConv(in_ch=in_hsz,
                                   out_ch=out_hsz,
                                   k=kernel_size,
                                   dim=1,
                                   relu=relu)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        return self.conv(x)  # (N, L, D)
        
'''
Core model
'''
class TRGAT(nn.Module):

    def __init__(self, opt):
        """
        Input:
            in_q_dim: dimension of input question, 
            in_i_dim: dimension of input image 
        Output:

        """
        super(TRGAT, self).__init__()
        self.opt = opt
        self.inference_mode = opt.inference_mode
        self.sub_flag = opt.sub_flag
        self.vfeat_flag = opt.vfeat_flag
        self.vcpt_flag = opt.vcpt_flag
        self.vfeat_size = opt.vfeat_size
        self.t_iter = opt.t_iter
        self.extra_span_length = opt.extra_span_length
        self.add_local = opt.add_local
        self.use_sup_att = opt.use_sup_att
        self.num_negatives = opt.num_negatives
        self.negative_pool_size = opt.negative_pool_size
        self.num_hard = opt.num_hard
        self.drop_topk = opt.drop_topk
        self.margin = opt.margin
        self.att_loss_type = opt.att_loss_type
        self.scale = opt.scale
        self.alpha = opt.alpha
        self.dropout = opt.dropout
        self.hsz = opt.hsz
        self.bsz = None
        self.num_seg = None
        self.num_a = 5
        self.flag_cnt = self.sub_flag + self.vfeat_flag

        self.wd_size = opt.embedding_size
        self.bridge_hsz = 300

        self.bert_word_encoding_fc = nn.Sequential(
            nn.LayerNorm(self.wd_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.wd_size, self.bridge_hsz),
            nn.ReLU(True),
            nn.LayerNorm(self.bridge_hsz),
        )

        self.str_attn = StructuredAttention(dropout=self.dropout,
                                            scale=opt.scale,
                                            add_void=opt.add_non_visual)  # no parameters inside

        self.c2q_down_projection = nn.Sequential(
            nn.LayerNorm(3 * self.hsz),
            nn.Dropout(self.dropout),
            nn.Linear(3*self.hsz, self.hsz),
            nn.ReLU(True),
        )
        self.cls_encoder = StackedEncoder(n_blocks=opt.cls_encoder_n_blocks,
                                          n_conv=opt.cls_encoder_n_conv,
                                          kernel_size=opt.cls_encoder_kernel_size,
                                          num_heads=opt.cls_encoder_n_heads,
                                          hidden_size=self.hsz,
                                          dropout=self.dropout)

        self.cls_projection_layers = nn.ModuleList(
            [
                LinearWrapper(in_hsz=self.hsz,
                              out_hsz=self.hsz,
                              layer_norm=True,
                              dropout=self.dropout,
                              relu=True)
            ] +
            [
                ConvLinear(in_hsz=self.hsz,
                           out_hsz=self.hsz,
                           kernel_size=3,
                           layer_norm=True,
                           dropout=self.dropout,
                           relu=True)
                for _ in range(self.t_iter)])

        self.temporal_scoring_st_layers = nn.ModuleList([
            LinearWrapper(in_hsz=self.hsz,
                          out_hsz=1,
                          layer_norm=True,
                          dropout=self.dropout,
                          relu=False)
            for _ in range(self.t_iter+1)])

        self.temporal_scoring_ed_layers = nn.ModuleList([
            LinearWrapper(in_hsz=self.hsz,
                          out_hsz=1,
                          layer_norm=True,
                          dropout=self.dropout,
                          relu=False)
            for _ in range(self.t_iter+1)])

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="sum")
        self.classifier = LinearWrapper(in_hsz=self.hsz * 2 if self.add_local else self.hsz,
                                        out_hsz=1,
                                        layer_norm=True,
                                        dropout=self.dropout,
                                        relu=False)

        if self.flag_cnt == 2:
            self.concat_fc = nn.Sequential(
                nn.LayerNorm(3 * self.hsz),
                nn.Linear(3 * self.hsz, self.hsz),
                nn.ReLU(True),
                nn.LayerNorm(self.hsz),
            )
            self.MultimodelAttention = nn.Sequential(
                nn.Linear(self.hsz, 1),
                nn.ReLU(True)
            )

        # self.Q_GCN = GAttNet(1, 1, self.hsz, self.hsz)
        # self.I_GCN = ImageGCN(self.embedding_dim, self.hsz, self.hsz)
        self.Implict_GAT = GAttNet(1, 1, self.hsz, self.hsz)
        self.Spatial_GAT = GAttNet(2, 11, self.hsz, self.hsz)
        # self.S_GCN = TextGCN(self.embedding_dim, self.hsz, self.out_feat_dim)
        # self.VideoSelfAttention = VideoSelfAttention(obj_num= opt.num_region, emb_dim= self.hsz)


        self.input_embedding = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.bridge_hsz, self.hsz),
            nn.ReLU(True),
            nn.LayerNorm(self.hsz),
        )

        self.input_encoder = StackedEncoder(n_blocks=opt.input_encoder_n_blocks,
                                            n_conv=opt.input_encoder_n_conv,
                                            kernel_size=opt.input_encoder_kernel_size,
                                            num_heads=opt.input_encoder_n_heads,
                                            hidden_size=self.hsz,
                                            dropout=self.dropout)
        
        if self.sub_flag:
            print("Activate sub branch")

        if self.vfeat_flag:
            print("Activate vid branch")
            self.vid_fc = nn.Sequential(
                nn.LayerNorm(self.vfeat_size),
                nn.Dropout(self.dropout),
                nn.Linear(self.vfeat_size, self.bridge_hsz),
                nn.ReLU(True),
                nn.LayerNorm(self.bridge_hsz)
            )
    def forward(self, batch):
        """
        Input: 
            que_emb: question embedding # (batch, num_words, word_emb_dim)
            vid_emb: video embedding # (batch, num_frames, num_regions, obj_emb_dim)
            ans_emb: answer embedding # (batch, num_choice, word_emb_dim)
            q_adj: adjent matrix of question # (batch, num_words, num_words)

        Output:
            out: (batch, score)
        """
        self.bsz = batch["qas_bert"].shape[0]
        bsz = self.bsz
        num_a = self.num_a
        hsz = self.hsz
    
        # Embedding qa-pair into low dimension
        qa_embed = self.base_encoder(batch["qas_bert"].view(bsz*num_a, -1, self.wd_size),  # (N*5, L, D)
                                    batch["qas_mask"].view(bsz * num_a, -1),  # (N*5, L)
                                    self.bert_word_encoding_fc, # init embedding (N*5, L, bridge_hsz)
                                    self.input_embedding, # Downsize encoder (N*5, L, hsz)
                                    self.input_encoder)  # (N*5, L, D)

        # qa_embed = self.advanced_encoder(qa_embed.contiguous().view(bsz, num_a, -1, self.hsz))
        # qa_embed_adv = qa_embed.view(bsz, num_a, -1, self.hsz)
        a_embed = qa_embed.view(bsz, num_a, 1, -1, hsz)  # (N, 5, 1, L, D)
        a_mask = batch["qas_mask"].view(bsz, num_a, 1, -1)  # (N, 5, 1, L)
        
        # Attention Output
        attended_sub, attended_vid, attended_vid_mask, attended_sub_mask = (None, ) * 4
        
        # Subtitle Attention
        if self.sub_flag:
            num_imgs, num_words = batch["sub_bert"].shape[1:3]

            ### Replace with GCN
            sub_embed = self.base_encoder(batch["sub_bert"].view(bsz*num_imgs, num_words, -1),  # (N*Li, Lw)
                                          batch["sub_mask"].view(bsz * num_imgs, num_words),  # (N*Li, Lw)
                                          self.bert_word_encoding_fc,
                                          self.input_embedding,
                                          self.input_encoder)  # (N*Li, Lw, D)
                                          
            # sub_embed = self.advanced_encoder(sub_embed.contiguous().view(bsz, num_imgs, num_words, -1))

            sub_embed = sub_embed.contiguous().view(bsz, 1, num_imgs, num_words, -1)  # (N, Li, Lr, D)
            sub_mask = batch["sub_mask"].view(bsz, 1, num_imgs, num_words)  # (N, 1, Li, Lr)

            attended_sub, attended_sub_mask, sub_raw_s, sub_normalized_s = \
                self.qa_ctx_attention(a_embed, sub_embed, a_mask, sub_mask,
                                      noun_mask=None,
                                      non_visual_vectors=None)

            other_outputs["sub_normalized_s"] = sub_normalized_s
            other_outputs["sub_raw_s"] = sub_raw_s
            
        # Video Frame Attention
        if self.vfeat_flag:
            num_imgs, num_regions = batch["vid"].shape[1:3]
            vid_embed = F.normalize(batch["vid"], p=2, dim=-1)  # (N, Li, Lr, D)

            vid_embed = self.base_encoder(vid_embed.view(bsz*num_imgs, num_regions, -1),  # (N*Li, Lw)
                                          batch["vid_mask"].view(bsz * num_imgs, num_regions),  # (N*Li, Lr)
                                          self.vid_fc,
                                          self.input_embedding,
                                          self.input_encoder)  # (N*Li, L, D)
                                
            # vid_embed = self.advanced_encoder(vid_embed.contiguous().view(bsz, num_imgs, num_regions, -1))
            vid_embed_adv = vid_embed.contiguous().view(bsz, num_imgs, num_regions, -1)  # (N, 1, Li, Lr, D)
            for idx in range(vid_embed_adv.shape[1]):
                adj_matrix = torch.ones(vid_embed_adv.shape[0], vid_embed_adv.shape[2], vid_embed_adv.shape[2], 1, requires_grad=True).to(self.opt.device)
                img_embed = self.Implict_GAT(vid_embed_adv[:, idx], adj_matrix)
                if idx == 0:
                    vid_embed = img_embed.unsqueeze(1)
                else:
                    vid_embed = torch.cat((vid_embed, img_embed.unsqueeze(1)), dim = 1)
            
            vid_embed = vid_embed.contiguous().view(bsz, num_imgs, num_regions, -1)  # (N, Li, Lr, D)
            # Residual connection
            vid_embed = vid_embed + vid_embed_adv
            # vid_embed = self.VideoSelfAttention(vid_embed)
            vid_embed = vid_embed.contiguous().view(bsz, 1, num_imgs, num_regions, -1)  # (N, 1, Li, Lr, D)
            vid_mask = batch["vid_mask"].view(bsz, 1, num_imgs, num_regions)  # (N, 1, Li, Lr)

            attended_vid, attended_vid_mask, vid_raw_s, vid_normalized_s = \
                self.qa_ctx_attention(a_embed, vid_embed, a_mask, vid_mask,
                                      noun_mask=None,
                                      non_visual_vectors=None)
            if self.vcpt_flag:
                num_imgs, num_regions = batch["vcpt_emb"].shape[1:3]
                vcpt_embed = F.normalize(batch["vcpt_emb"], p=2, dim=-1)  # (N, Li, Lr, D)
                # vcpt_embed
                vcpt_embed = self.base_encoder(vcpt_embed.view(bsz*num_imgs, num_regions, -1),  # (N*Li, Lw)
                                              batch["vcpt_mask"].view(bsz * num_imgs, num_regions),  # (N*Li, Lr)
                                              self.vid_fc,
                                              self.input_embedding,
                                              self.input_encoder)  # (N*Li, L, D)
                                    
                # vcpt_embed = self.advanced_encoder(vcpt_embed.contiguous().view(bsz, num_imgs, num_regions, -1))
                vcpt_embed_adv = vcpt_embed.contiguous().view(bsz, num_imgs, num_regions, -1)  # (N, 1, Li, Lr, D)
                for idx in range(vcpt_embed_adv.shape[1]):
                    adj_matrix = batch["spa_adj_mtx"][:, idx]
                    img_embed = self.Spatial_GAT(vcpt_embed_adv[:, idx], adj_matrix)
                    if idx == 0:
                        vcpt_embed = img_embed.unsqueeze(1)
                    else:
                        vcpt_embed = torch.cat((vcpt_embed, img_embed.unsqueeze(1)), dim = 1)

                vcpt_embed = vcpt_embed.contiguous().view(bsz, num_imgs, num_regions, -1)  # (N, Li, Lr, D)
                # Residual connection
                vcpt_embed = vcpt_embed + vcpt_embed_adv
                # vcpt_embed = self.VideoSelfAttention(vcpt_embed)
                vcpt_embed = vcpt_embed.contiguous().view(bsz, 1, num_imgs, num_regions, -1)  # (N, 1, Li, Lr, D)
                vcpt_mask = batch["vcpt_mask"].view(bsz, 1, num_imgs, num_regions)  # (N, 1, Li, Lr)

                attended_vcpt, attended_vcpt_mask, vcpt_raw_s, vcpt_normalized_s = \
                    self.qa_ctx_attention(a_embed, vcpt_embed, a_mask, vcpt_mask,
                                          noun_mask=None,
                                          non_visual_vectors=None)


        # If vfeat_flag && sub_flag
        if self.flag_cnt == 2:
            """
            三个依次级联，再加入一个点乘的似乎效果不太好，感觉应该尝试三个级联+attention
            """
            """
            加了attention效果也不好 我吐了(可能是没加softmax？加一下试试) 加了softmax好像差不多了，但是要低一点点。妈的 写错了 妈的 又写错了！
            """
            
            # visual_text_embedding = torch.cat([attended_sub,
            #                                    attended_vid,
            #                                    attended_sub * attended_vid], dim=-1)  # (N, 5, Li, Lqa, 3D)
            visual_text_embedding = torch.stack([attended_sub, attended_vid, attended_vcpt], dim = -2)
            v_shape_0, v_shape_1, v_shape_2, v_shape_3 = visual_text_embedding.shape[:4]
            visual_text_embedding = F.softmax(self.MultimodelAttention(visual_text_embedding), dim = -2) * 3 * visual_text_embedding
            visual_text_embedding = visual_text_embedding.view(v_shape_0, v_shape_1, v_shape_2, v_shape_3, -1)

            visual_text_embedding = self.concat_fc(visual_text_embedding)  # (N, 5, Li, Lqa, D)
            out, target, t_scores = self.classfier_head_multi_proposal(
                visual_text_embedding, attended_vid_mask, batch["target"], batch["ts_label"], batch["ts_label_mask"],
                extra_span_length=self.extra_span_length)
        # Subtitle only
        elif self.sub_flag:
            out, target, t_scores = self.classfier_head_multi_proposal(
                attended_sub, attended_sub_mask, batch["target"], batch["ts_label"], batch["ts_label_mask"],
                extra_span_length=self.extra_span_length)
        
        # Video only
        elif self.vfeat_flag:
            out, target, t_scores = self.classfier_head_multi_proposal(
                attended_vid, attended_vid_mask, batch["target"], batch["ts_label"], batch["ts_label_mask"],
                extra_span_length=self.extra_span_length)
        else:
            raise NotImplementedError
        assert len(out) == len(target)
        # print(self.inference_mode)
        if self.inference_mode:
            return out, F.softmax(t_scores, dim=2)
            
        att_loss = torch.zeros(1).to(self.opt.device)
        att_predictions = None
        # if (self.use_sup_att or not self.training) and self.vfeat_flag:
        if self.use_sup_att and self.training and self.vfeat_flag:
            start_indices = batch["anno_st_idx"]
            try:
                cur_att_loss, cur_att_predictions = \
                    self.get_att_loss(vid_raw_s, batch["att_labels"], batch["target"], batch["qas"],
                                      qids=batch["qid"],
                                      q_lens=batch["q_l"],
                                      vid_names=batch["vid_name"],
                                      img_indices=batch["image_indices"],
                                      boxes=batch["boxes"],
                                      start_indices=start_indices,
                                      num_negatives=self.num_negatives,
                                      use_hard_negatives=batch["use_hard_negatives"],
                                      drop_topk=self.drop_topk)
            except AssertionError as e:
                save_pickle(
                    {"batch": batch, "start_indices": start_indices, "vid_raw_s": other_outputs["vid_raw_s"]},
                    "err_dict.pickle"
                )
                import sys
                sys.exit(1)
            att_loss += cur_att_loss
            att_predictions = cur_att_predictions
        # print("att:",att_loss)
        temporal_loss = self.get_ts_loss(temporal_scores=t_scores,
                                         ts_labels=batch["ts_label"],
                                         answer_indices=batch["target"])

        if self.training:
            return [out, target], att_loss, att_predictions, temporal_loss, t_scores, other_outputs
        else:
            return out, att_loss, att_predictions, temporal_loss, F.softmax(t_scores, dim=2), other_outputs
    pass

    @classmethod
    def advanced_encoder(self, inputs):

        batch_size, img_num, obj_num, emb_dim = inputs.shape[:]
        inputs_view = inputs.view(batch_size, img_num, -1)
        inputs_trans = inputs_view.permute(0, 2, 1)
        self_weight = F.softmax(torch.bmm(inputs_view, inputs_trans), dim = 1)

        #learnable_weight = F.softmax(self.Linear(inputs_view), dim = 1)
        #learnable_weight_trans = learnable_weight.permute(0,2,1)
        #learnable_weight_metrix = torch.bmm(learnable_weight, learnable_weight_trans)

        out = torch.bmm(self_weight, inputs_view)
        #out = torch.bmm(learnable_weight_metrix, out)
        output = out.view(batch_size, img_num, obj_num, emb_dim)

        self_weight_sum = torch.sum(self_weight, dim = 2)
        self_weight_idx = torch.argmax(self_weight_sum, dim = 1)
        self_weight_mask = torch.zeros(batch_size,img_num).cuda()
        for idx, i in enumerate(self_weight_idx):
            self_weight_mask[idx][i]=1
        self_weight_mask = self_weight_mask.unsqueeze(-1).unsqueeze(-1)
        output_max = torch.sum(output*self_weight_mask, dim = 1)


        return output
        pass
    @classmethod
    def base_encoder(cls, data, data_mask, init_encoder, downsize_encoder, input_encoder):
        """ Raw data --> higher-level embedding
        Args:
            data: (N, L) for text, (N, L, D) for video
            data_mask: (N, L)
            init_encoder: word_embedding layer for text, MLP (downsize) for video
            downsize_encoder: MLP, down project to hsz
            input_encoder: multiple layer of encoder block, with residual connection, CNN, layernorm, etc
        Returns:
            encoded_data: (N, L, D)
        """
        data = downsize_encoder(init_encoder(data))
        return input_encoder(data, data_mask)

    def qa_ctx_attention(self, qa_embed, ctx_embed, qa_mask, ctx_mask, noun_mask, non_visual_vectors):
        """ Align image regions with QA words
        Args:
            qa_embed: (N, 5, 1, Lqa, D)
            qa_mask:  (N, 5, 1, Lqa)
            ctx_embed: (N, 1, Li, Lr, D)
            ctx_mask: (N, 1, Li, Lr)
            noun_mask: (N, 5, Lqa)
            non_visual_vectors: (m, D), m is a tunable parameter
        Returns:
        """
        num_img, num_region = ctx_mask.shape[2:]

        u_a, raw_s, s_mask, s_normalized = self.str_attn(
            qa_embed, ctx_embed, qa_mask, ctx_mask,
            noun_mask=noun_mask, void_vector=non_visual_vectors)  # (N, 5, Li, Lqa, D), (N, 5, Li, Lqa, lr) x2
        qa_embed = qa_embed.repeat(1, 1, num_img, 1, 1)
        mixed = torch.cat([qa_embed,
                           u_a,
                           qa_embed*u_a], dim=-1)  # (N, 5, Li, Lqa, D)
        mixed = self.c2q_down_projection(mixed)  # (N, 5, Li, Lqa, D)
        mixed_mask = (s_mask.sum(-1) != 0).float()  # (N, 5, Li, Lqa)
        return mixed, mixed_mask, raw_s, s_normalized

    def residual_temporal_predictor(self, layer_idx, input_tensor):
        """
        Args:
            layer_idx (int):
            input_tensor: (N, L, D)

        Returns:
            temporal_score
        """
        input_tensor = input_tensor + self.cls_projection_layers[layer_idx](input_tensor)  # (N, L, D)
        t_score_st = self.temporal_scoring_st_layers[layer_idx](input_tensor)  # (N, L, 1)
        t_score_ed = self.temporal_scoring_ed_layers[layer_idx](input_tensor)  # (N, L, 1)
        t_score = torch.cat([t_score_st, t_score_ed], dim=2)  # (N, L, 2)
        return input_tensor, t_score

    def get_ts_loss(self, temporal_scores, ts_labels,  answer_indices):
        """
        Args:
            temporal_scores: (N, 5, Li, 2)
            ts_labels: dict(st=(N, ), ed=(N, ))
            answer_indices: (N, )

        Returns:

        """
        bsz = len(answer_indices)
        # compute loss
        ca_temporal_scores_st_ed = \
            temporal_scores[torch.arange(bsz, dtype=torch.long), answer_indices]  # (N, Li, 2)
        loss_st = self.temporal_criterion(ca_temporal_scores_st_ed[:, :, 0], ts_labels["st"])
        loss_ed = self.temporal_criterion(ca_temporal_scores_st_ed[:, :, 1], ts_labels["ed"])
        return (loss_st + loss_ed) / 2.
    
    def classfier_head_multi_proposal(self, statement, statement_mask, targets, ts_labels, ts_labels_mask,
                                      max_num_proposal=1, ce_prob_thd=0.01, iou_thd=0.5, extra_span_length=3):
        """Predict the probabilities of each statements being true. Statements = QA + Context.
        Args:
            statement: (N, 5, Li, Lqa, D)
            statement_mask: (N, 5, Li, Lqa)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            ts_labels_mask: (N, Li)
            max_num_proposal (int):
            ce_prob_thd (float): threshold for p1*p2 (st, ed)
            iou_thd (float): threshold for temporal iou
            extra_span_length (int): expand the localized span to give a little bit extra context
        Returns:
        """
        bsz, num_a, num_img, num_words = statement_mask.shape
        statement = statement.view(bsz*num_a*num_img, num_words, -1)  # (N*5*Li, Lqa, D)
        statement_mask = statement_mask.view(bsz*num_a*num_img, num_words)  # (N*5*Li, Lqa)
        statement = self.cls_encoder(statement, statement_mask)  # (N*5*Li, Lqa, D)
        max_statement = torch.max(mask_logits(statement, statement_mask.unsqueeze(2)), 1)[0]  # (N*5*Li, D)
        max_statement_mask = (statement_mask.sum(1) != 0).float().view(bsz, num_a, num_img, 1)  # (N, 5, Li, 1)
        max_statement = max_statement.view(bsz*num_a, num_img, -1)  # (N, 5, Li, D)

        t_score_container = []
        encoded_max_statement_container = []
        encoded_max_statement = max_statement  # (N*5, Li, D)
        for layer_idx in range(self.t_iter+1):
            encoded_max_statement, prev_t_score = \
                self.residual_temporal_predictor(layer_idx, encoded_max_statement)
            t_score_container.append(prev_t_score.view(bsz, num_a, num_img, 2))  # (N, 5, Li, 2)
            encoded_max_statement_container.append(encoded_max_statement)  # (N*5, Li, D)
        if self.t_iter > 0:
            temporal_scores_st_ed = 0.5 * (t_score_container[0] + torch.stack(t_score_container[:1]).mean(0))
        else:
            temporal_scores_st_ed = t_score_container[0]  # (N, 5, Li, 2)

        # mask before softmax
        temporal_scores_st_ed = mask_logits(temporal_scores_st_ed, ts_labels_mask.view(bsz, 1, num_img, 1))

        # when predict answer, only consider 1st level representation !!!
        # since the others are all generated from the 1st level
        stacked_max_statement = encoded_max_statement_container[0].view(bsz, num_a, num_img, -1)   # (N, 5, Li, D)
        if self.add_local:
            max_max_statement, targets = self.get_proposals(
                stacked_max_statement, max_statement_mask, temporal_scores_st_ed,
                targets, ts_labels, max_num_proposal=max_num_proposal, iou_thd=iou_thd,
                ce_prob_thd=ce_prob_thd, extra_span_length=extra_span_length)   # (N, 5, D)
        else:
            max_max_statement = \
                    torch.max(mask_logits(stacked_max_statement, max_statement_mask), 2)[0]  # (N, 5, D)

        answer_scores = self.classifier(max_max_statement).squeeze(2)  # (N, 5)
        answer_scores += 1e-10
        return answer_scores, targets, temporal_scores_st_ed  # (N_new, 5), (N_new, ) (N, 5, Li, 2)

    def get_proposals(self, max_statement, max_statement_mask, temporal_scores,
                      targets, ts_labels, max_num_proposal=1, iou_thd=0.5, ce_prob_thd=0.01,
                      extra_span_length=3):
        """
        Args:
            max_statement: (N, 5, Li, D)
            max_statement_mask: (N, 5, Li, 1)
            temporal_scores: (N, 5, Li, 2)
            targets: (N, )
            ts_labels: (N, Li) for frm or N * (st, ed) for st_ed
            max_num_proposal:
            iou_thd:
            ce_prob_thd:
            extra_span_length:
        Returns:

        """
        bsz, num_a, num_img, _ = max_statement_mask.shape
        if self.training:
            ca_temporal_scores_st_ed = \
                temporal_scores[torch.arange(bsz, dtype=torch.long), targets].data  # (N, Li, 2)
            ca_temporal_scores_st_ed = F.softmax(ca_temporal_scores_st_ed, dim=1)  # (N, Li, 2)
            ca_pred_spans = find_max_triples(ca_temporal_scores_st_ed[:, :, 0],
                                             ca_temporal_scores_st_ed[:, :, 1],
                                             topN=max_num_proposal,
                                             prob_thd=ce_prob_thd)  # N * [(st_idx, ed_idx, confidence), ...]
            # +1 for ed index before forward into get_high_iou_spans func.
            ca_pred_spans = [[[sub_e[0], sub_e[1] + 1, sub_e[2]] for sub_e in e] for e in ca_pred_spans]
            spans = get_high_iou_sapns(zip(ts_labels["st"].tolist(), (ts_labels["ed"] + 1).tolist()),
                                       ca_pred_spans, iou_thd=iou_thd, add_gt=True)  # N * [(st, ed), ...]
            local_max_max_statement_list = []  # N_new * (5, D)
            global_max_max_statement_list = []  # N_new * (5, D)
            span_targets = []  # N_new * (1,)
            for idx, (t, span_sublist) in enumerate(zip(targets, spans)):
                span_targets.extend([t] * len(span_sublist))
                cur_global_max_max_statement = \
                    torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 1)[0]
                global_max_max_statement_list.extend([cur_global_max_max_statement] * len(span_sublist))
                for span in span_sublist:
                    span = expand_span(span, expand_length=extra_span_length)
                    cur_span_max_statement = mask_logits(
                        max_statement[idx, :, span[0]:span[1]],
                        max_statement_mask[idx, :, span[0]:span[1]])  # (5, Li[st:ed], D)
                    local_max_max_statement_list.append(torch.max(cur_span_max_statement, 1)[0])  # (5, D)
            local_max_max_statement = torch.stack(local_max_max_statement_list)  # (N_new, 5, D)
            global_max_max_statement = torch.stack(global_max_max_statement_list)  # (N_new, 5, D)
            max_max_statement = torch.cat([
                local_max_max_statement,
                global_max_max_statement], dim=-1)  # (N_new, 5, 2D)
            return max_max_statement, targets.new_tensor(span_targets)  # (N_new, 5, 2D), (N_new, )
        else:  # testing
            temporal_scores_st_ed = F.softmax(temporal_scores, dim=2)  # (N, 5, Li, 2)
            temporal_scores_st_ed_reshaped = temporal_scores_st_ed.view(bsz * num_a, -1, 2)  # (N*5, Li, 2)
            pred_spans = find_max_triples(temporal_scores_st_ed_reshaped[:, :, 0],
                                          temporal_scores_st_ed_reshaped[:, :, 1],
                                          topN=1, prob_thd=None)  # (N*5) * [(st, ed, confidence), ]
            pred_spans = flat_list_of_lists(pred_spans)  # (N*5) * (st, ed, confidence)
            pred_spans = torch.FloatTensor(pred_spans).to(temporal_scores_st_ed_reshaped.device)  # (N*5, 3)
            pred_spans, pred_scores = pred_spans[:, :2].long(), pred_spans[:, 2]  # (N*5, 2), (N*5, )
            pred_spans = [[e[0], e[1] + 1] for e in pred_spans]
            max_statement = max_statement.view(bsz * num_a, num_img, -1)  # (N*5, Li, D)
            max_statement_mask = max_statement_mask.view(bsz * num_a, num_img, -1)  # (N*5, Li, 1)
            local_max_max_statement_list = []  # N*5 * (D, )
            global_max_max_statement_list = []  # N*5 * (D, )
            for idx, span in enumerate(pred_spans):
                span = expand_span(span, expand_length=extra_span_length)
                cur_global_max_max_statement = \
                    torch.max(mask_logits(max_statement[idx], max_statement_mask[idx]), 0)[0]
                global_max_max_statement_list.append(cur_global_max_max_statement)
                cur_span_max_statement = mask_logits(
                    max_statement[idx, span[0]:span[1]],
                    max_statement_mask[idx, span[0]:span[1]])  # (Li[st:ed], D), words for span[0] == span[1]
                local_max_max_statement_list.append(torch.max(cur_span_max_statement, 0)[0])  # (D, )
            local_max_max_statement = torch.stack(local_max_max_statement_list)  # (N*5, D)
            global_max_max_statement = torch.stack(global_max_max_statement_list)  # (N*5, D)
            max_max_statement = torch.cat([
                local_max_max_statement,
                global_max_max_statement], dim=-1)  # (N_new, 5, 2D)
            return max_max_statement.view(bsz, num_a, -1), targets  # (N, 5, 2D), (N, )

    @classmethod
    def sample_negatives(cls, pred_score, pos_indices, neg_indices, num_negatives=2,
                         use_hard_negatives=False, negative_pool_size=0, num_hard=2, drop_topk=0):
        """ Sample negatives from a set of indices. Several sampling strategies are supported:
        1, random; 2, hard negatives; 3, drop_topk hard negatives; 4, mix easy and hard negatives
        5, sampling within a pool of hard negatives; 6, sample across images of the same video.
        Args:
            pred_score: (num_img, num_words, num_region)
            pos_indices: (N_pos, 3) all positive region indices for the same word, not necessaryily the same image.
            neg_indices: (N_neg, 3) ...
            num_negatives (int):
            use_hard_negatives (bool):
            negative_pool_size (int):
            num_hard (int):
            drop_topk (int):
        Returns:

        """
        num_unique_pos = len(pos_indices)
        sampled_pos_indices = torch.cat([pos_indices] * num_negatives, dim=0)
        if use_hard_negatives:
            # print("using use_hard_negatives")
            neg_scores = pred_score[neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]  # TODO
            max_indices = torch.sort(neg_scores, descending=True)[1].tolist()
            if negative_pool_size > num_negatives:  # sample from a pool of hard negatives
                hard_pool = max_indices[drop_topk:drop_topk + negative_pool_size]
                hard_pool_indices = neg_indices[hard_pool]
                num_hard_negs = num_negatives
                sampled_easy_neg_indices = []
                if num_hard < num_negatives:
                    easy_pool = max_indices[drop_topk + negative_pool_size:]
                    easy_pool_indices = neg_indices[easy_pool]
                    num_hard_negs = num_hard
                    num_easy_negs = num_negatives - num_hard_negs
                    sampled_easy_neg_indices = easy_pool_indices[
                        torch.randint(low=0, high=len(easy_pool_indices),
                                      size=(num_easy_negs * num_unique_pos, ), dtype=torch.long)
                    ]
                sampled_hard_neg_indices = hard_pool_indices[
                    torch.randint(low=0, high=len(hard_pool_indices),
                                  size=(num_hard_negs * num_unique_pos, ), dtype=torch.long)
                ]

                if len(sampled_easy_neg_indices) != 0:
                    sampled_neg_indices = torch.cat([sampled_hard_neg_indices, sampled_easy_neg_indices], dim=0)
                else:
                    sampled_neg_indices = sampled_hard_neg_indices

            else:  # directly take the top negatives
                sampled_neg_indices = neg_indices[max_indices[drop_topk:drop_topk+len(sampled_pos_indices)]]
        else:
            sampled_neg_indices = neg_indices[
                torch.randint(low=0, high=len(neg_indices), size=(len(sampled_pos_indices),), dtype=torch.long)
            ]
        return sampled_pos_indices, sampled_neg_indices

    def get_att_loss(self, scores, att_labels, target, words, vid_names, qids, q_lens, img_indices, boxes,
                     start_indices, num_negatives=2, use_hard_negatives=False, drop_topk=0):
        """ compute ranking loss, use for loop to find the indices,
        use advanced indexing to perform the real calculation
        Build a list contains a quaduple

        Args:
            scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
            att_labels: list(tensor), each has dimension (#num_imgs, #num_words, #regions), not batched
            target: 1D tensor (N, )
            words: LongTensor (N, 5, Lqa)
            vid_names: list(str) (N,)
            qids: list(int), (N, )
            q_lens: list(int), (N, )
            img_indices: list(list(int)), (N, Li), or None
            boxes: list(list(box)) of length N, each sublist represent an image,
                each box contains the coordinates of xyxy, or None
            num_negatives: number of negatives for each positive region
            use_hard_negatives: use hard negatives, uselect negatives with high scores
            drop_topk: drop topk highest negatives (since the top negatives might be correct, they are just not labeled)
            start_indices (list of int): each element is an index (at 0.5fps) of the first image
                with spatial annotation. If with_ts, set to zero
        Returns:
            att_loss: loss value for the batch
            att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
        """
        pos_container = []  # contains tuples of 5 elements, which are (batch_i, ca_i, img_i, word_i, region_i)
        neg_container = []
        for batch_idx in range(len(target)):  # batch
            ca_idx = target[batch_idx].cpu().item()
            gt_score = att_labels[batch_idx]  # num_img * (num_words, num_region)
            start_idx = start_indices[batch_idx]  # int
            num_img = len(gt_score)
            sen_l, _ = gt_score[0].shape
            pred_score = scores[batch_idx, ca_idx, :num_img, :sen_l]  # (num_img, num_words, num_region)

            # find positive and negative indices
            batch_pos_indices = []
            batch_neg_indices = []
            for img_idx, img_gt_score in enumerate(gt_score):
                img_idx = start_idx + img_idx
                img_pos_indices = torch.nonzero(img_gt_score)  # (N_pos, 2) ==> (#words, #regions)
                if len(img_pos_indices) == 0:  # skip if no positive indices
                    continue
                img_pos_indices = torch.cat([img_pos_indices.new_full([len(img_pos_indices), 1], img_idx),
                                             img_pos_indices], dim=1)  # (N_pos, 3) ==> (#img, #words, #regions)

                img_neg_indices = torch.nonzero(img_gt_score == 0)  # (N_neg, 2)
                img_neg_indices = torch.cat([img_neg_indices.new_full([len(img_neg_indices), 1], img_idx),
                                             img_neg_indices], dim=1)  # (N_neg, 3)

                batch_pos_indices.append(img_pos_indices)
                batch_neg_indices.append(img_neg_indices)

            if len(batch_pos_indices) == 0:  # skip if empty ==> no gt label for the video
                continue
            batch_pos_indices = torch.cat(batch_pos_indices, dim=0)  # (N_pos, 3) -->
            batch_neg_indices = torch.cat(batch_neg_indices, dim=0)  # (N_neg, 3)

            # sample positives and negatives
            available_img_indices = batch_pos_indices[:, 0].unique().tolist()
            for img_idx in available_img_indices:
                # pos_indices for a certrain img
                img_idx_pos_indices = batch_pos_indices[batch_pos_indices[:, 0] == img_idx]
                img_idx_neg_indices = batch_neg_indices[batch_neg_indices[:, 0] == img_idx]
                available_word_indices = img_idx_pos_indices[:, 1].unique().tolist()
                for word_idx in available_word_indices:
                    # positives and negatives for a given image-word pair, specified by img_idx-word_idx
                    img_idx_word_idx_pos_indices = img_idx_pos_indices[img_idx_pos_indices[:, 1] == word_idx]
                    img_idx_word_idx_neg_indices = img_idx_neg_indices[img_idx_neg_indices[:, 1] == word_idx]
                    # actually all the positives, not sampled pos
                    sampled_pos_indices, sampled_neg_indices = \
                        self.sample_negatives(pred_score,
                                              img_idx_word_idx_pos_indices, img_idx_word_idx_neg_indices,
                                              num_negatives=num_negatives, use_hard_negatives=use_hard_negatives,
                                              negative_pool_size=self.negative_pool_size,
                                              num_hard=self.num_hard, drop_topk=drop_topk)

                    base_indices = torch.LongTensor([[batch_idx, ca_idx]] * len(sampled_pos_indices)).\
                        to(sampled_pos_indices.device)
                    pos_container.append(torch.cat([base_indices, sampled_pos_indices], dim=1))
                    neg_container.append(torch.cat([base_indices, sampled_neg_indices], dim=1))

        pos_container = torch.cat(pos_container, dim=0)
        neg_container = torch.cat(neg_container, dim=0)

        # contain all the predictions and gt labels in this batch, only consider the ones with gt labels
        # also only consider the positive answer.
        att_predictions = None
        if not self.training and self.vfeat_flag:
            att_predictions = dict(det_q=[],
                                   det_ca=[])
            unique_pos_container = np.unique(pos_container.cpu().numpy(), axis=0)  # unique rows in the array
            for row in unique_pos_container:
                batch_idx, ca_idx, img_idx, word_idx, region_idx = row
                start_idx = start_indices[batch_idx]  # int
                cur_q_len = q_lens[batch_idx]
                num_region = att_labels[batch_idx][img_idx-start_idx].shape[1]  # num_img * (num_words, num_region)
                if len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()) != \
                        len(boxes[batch_idx][img_idx-start_idx]):
                    print("scores[batch_idx, ca_idx, img_idx, word_idx].data.cpu()",
                          len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()))
                    print("len(boxes[batch_idx][img_idx-start_idx])", len(boxes[batch_idx][img_idx-start_idx]))
                    print("boxes, batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx",
                          batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx)
                    print(row)
                    raise AssertionError
                cur_det_data = {
                        "pred": scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu(),
                        "word": words[batch_idx, ca_idx, word_idx],
                        "qid": qids[batch_idx],
                        "vid_name": vid_names[batch_idx],
                        "img_idx": img_indices[batch_idx][img_idx],  # full indices
                        "boxes": boxes[batch_idx][img_idx-start_idx]  # located boxes
                    }
                if word_idx < cur_q_len:
                    att_predictions["det_q"].append(cur_det_data)
                else:
                    att_predictions["det_ca"].append(cur_det_data)

        pos_scores = scores[pos_container[:, 0], pos_container[:, 1], pos_container[:, 2],
                            pos_container[:, 3], pos_container[:, 4]]
        neg_scores = scores[neg_container[:, 0], neg_container[:, 1], neg_container[:, 2],
                            neg_container[:, 3], neg_container[:, 4]]

        if self.att_loss_type == "hinge":
            # max(0, m + S_pos - S_neg)
            att_loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0).sum()
        elif self.att_loss_type == "lse":
            # log[1 + exp(scale * (S_pos - S_neg))]
            att_loss = torch.log1p(torch.exp(self.alpha * (neg_scores - pos_scores))).sum()
        else:
            raise NotImplementedError("Only support hinge and lse")
        return att_loss, att_predictions

    def get_att_prediction(self, scores, object_vocab, words, vid_names, qids, img_indices, boxes,
                           start_indices, score_thd=0.2):
        """ compute ranking loss, use for loop to find the indices,
        use advanced indexing to perform the real calculation
        Build a list contains a quaduple

        Args:
            scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
            object_vocab: list, object word ids in the vocabulary
            words: LongTensor (N, 5, Lqa)
            vid_names: list(str) (N,)
            qids: list(int), (N, )
            img_indices: list(list(int)), (N, Li), or None
            boxes: list(list(box)) of length N, each sublist represent an image,
                each box contains the coordinates of xyxy, or None
            start_indices (list of int): each element is an index (at 0.5fps) of the first image
                with spatial annotation. If with_ts, set to zero
            score_thd: only keep boxes with score higher than this value
        Returns:
            att_loss: loss value for the batch
            att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
        """
        # contain all the predictions and gt labels in this batch, only consider the ones with gt labels
        # also only consider the positive answer.
        att_predictions = None
        if self.vfeat_flag:
            att_predictions = []
            for batch_idx in range(len(scores)):
                start_idx = start_indices[batch_idx]  # int
                q_att_predictions = dict()  # predictions associated with this question
                for ans_idx in range(5):
                    q_att_predictions[ans_idx] = []
                    for img_idx_local in range(len(boxes[batch_idx])):
                        # img_idx_local: for the imgs with box anno
                        # img_idx_global: for all the imgs, including ones without box anno
                        img_idx_global = img_idx_local + start_idx
                        cur_img_scores = scores[batch_idx, ans_idx, img_idx_global]  # (Lqa, Lr)
                        cur_words = words[batch_idx, ans_idx].tolist()  # (Lqa, )
                        cur_img_boxes = boxes[batch_idx][img_idx_local]
                        for word_idx, w in enumerate(cur_words):
                            if w in object_vocab:
                                cur_word_region_scores = cur_img_scores[word_idx].data.cpu().numpy()  # (Lr, )
                                accepted_region_ids = np.nonzero(cur_word_region_scores >= score_thd)[0].tolist()
                                accepted_region_scores = [float(cur_word_region_scores[i]) for i in accepted_region_ids]
                                accepted_region_boxes = [cur_img_boxes[i] for i in accepted_region_ids]
                                sorted_indices = np.argsort(accepted_region_scores)
                                accepted_region_scores = [accepted_region_scores[i] for i in sorted_indices]
                                accepted_region_boxes = [accepted_region_boxes[i] for i in sorted_indices]
                                cur_det_data = {
                                    "pred": accepted_region_scores,
                                    "bbox": accepted_region_boxes,
                                    "word": int(words[batch_idx, ans_idx, word_idx]),
                                    "qid": int(qids[batch_idx]),
                                    "vid_name": vid_names[batch_idx],
                                    "img_idx": img_indices[batch_idx][img_idx_global],  # image file name id
                                }
                                q_att_predictions[ans_idx].append(cur_det_data)
                att_predictions.append(q_att_predictions)
        return att_predictions


class VideoSelfAttention(nn.Module):
    def __init__(self, obj_num = 25, emb_dim = 128):
        super(VideoSelfAttention, self).__init__()
        self.input_dim = obj_num * emb_dim
        self.Linear = nn.Sequential(
                            nn.ReLU(),
                            nn.LayerNorm(self.input_dim),
                            nn.Linear(self.input_dim, self.input_dim // 2),
                            nn.ReLU(),
                            nn.Linear(self.input_dim // 2, 1),

        )
        pass
    def forward(self, inputs):
        batch_size, img_num, obj_num, emb_dim = inputs.shape[:]
        inputs_view = inputs.view(batch_size, img_num, -1)
        inputs_trans = inputs_view.permute(0, 2, 1)
        self_weight = F.softmax(torch.bmm(inputs_view, inputs_trans), dim = 1)

        learnable_weight = F.softmax(self.Linear(inputs_view), dim = 1)
        learnable_weight_trans = learnable_weight.permute(0,2,1)
        learnable_weight_metrix = torch.bmm(learnable_weight, learnable_weight_trans)

        out = torch.bmm(self_weight, inputs_view)
        out = torch.bmm(learnable_weight_metrix, out)
        output = out.view(batch_size, img_num, obj_num, emb_dim)

        return output

class SubSelfAttention(nn.Module):
    def __init__(self):
        super(SubSelfAttention, self).__init__()
        #self.Linear = nn.Linear(obj_num * emb_dim, 1)

    def forward(self, inputs):
        batch_size, img_num, word_num, emb_dim = inputs.shape[:]
        inputs_view = inputs.view(batch_size, img_num, -1)
        inputs_trans = inputs_view.permute(0, 2, 1)
        self_weight = F.softmax(torch.bmm(inputs_view, inputs_trans), dim = 1)
        #learnable_weight = F.softmax(self.Linear(inputs_view), dim = 1).squeeze()
        out = torch.bmm(self_weight, inputs_view)
        #out = torch.bmm(learnable_weight, self_weight)
        output = out.view(batch_size, img_num, word_num, emb_dim)
        self_weight_sum = torch.sum(self_weight, dim = 2)
        self_weight_idx = torch.argmax(self_weight_sum, dim = 1)
        self_weight_mask = torch.zeros(batch_size,img_num).cuda()
        for idx, i in enumerate(self_weight_idx):
            self_weight_mask[idx][i]=1
        self_weight_mask = self_weight_mask.unsqueeze(-1).unsqueeze(-1)
        output_max = torch.sum(output*self_weight_mask, dim = 1)

        return output, self_weight, output_max

class QAGuidedAttention(nn.Module):
    def __init__(self, in_feat_dim, hidden_dim):
        super(QAGuidedAttention, self).__init__()
        self.Linear = nn.Linear(in_feat_dim, hidden_dim)

    def forward(self, qa_emb, img_emb):
        #img_emb_trans = img_emb.permute(0, 2, 1)
        learnable_weight = F.softmax(self.Linear(img_emb), dim = 1)
        learnable_weight_trans = learnable_weight.permute(0, 2, 1)
        learnable_weight_metrix = torch.bmm(learnable_weight,learnable_weight_trans)
        img_emb_trans = torch.bmm(learnable_weight_metrix, img_emb).permute(0, 2, 1)
        weight = F.softmax(torch.bmm(qa_emb, img_emb_trans), dim = 1)
        out = torch.bmm(weight, img_emb)
        return out
        pass

'''
Toy GCN for question embedding
'''
class TextGCN(nn.Module):

    def __init__(self, in_feat_dim, hidden_dim, out_feat_dim):
        super(TextGCN, self).__init__()
        
        self.in_feat_dim = in_feat_dim
        self.hidden_dim = hidden_dim
        self.out_feat_dim = out_feat_dim
        self.GCN1 = GCNLayer(self.in_feat_dim, self.hidden_dim)
        self.GCN2 = GCNLayer(self.hidden_dim, self.out_feat_dim)
        self.Linear = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.LayerNorm(self.in_feat_dim * 2),
                            nn.Linear(self.in_feat_dim + self.in_feat_dim, 1)
        )
        pass
    pass

    def forward(self, features):
        obj_num = features.shape[1]
        features_exp = features.unsqueeze(1)
        features_exp = features_exp.expand(-1, obj_num, -1, -1)
        features_exp_permute = features_exp.permute(0, 2, 1, 3)
        features_concat = torch.cat([features_exp, features_exp_permute], dim = -1)
        adj = self.Linear(features_concat).squeeze()

        features = features.squeeze()
        GCN_out_1 = self.GCN1(adj, features)
        relu_GCN_out_1 = F.relu(GCN_out_1)
        GCN_out_2 = self.GCN2(adj, relu_GCN_out_1)

        return GCN_out_2
        pass


class ImageGCN(nn.Module):

    def __init__(self, in_feat_dim, hidden_dim, out_feat_dim):
        super(ImageGCN, self).__init__()
        
        self.in_feat_dim = in_feat_dim
        self.hidden_dim = hidden_dim
        self.out_feat_dim = out_feat_dim
        self.GCN1 = GCNLayer(self.in_feat_dim, self.hidden_dim)
        self.GCN2 = GCNLayer(self.hidden_dim, self.out_feat_dim)
        self.Linear = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Linear(self.in_feat_dim + self.in_feat_dim, self.hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.LayerNorm(self.hidden_dim),
                            nn.Linear(self.hidden_dim, 1)
        )
        pass
    pass

    def forward(self, features):
        
        # Build Adjent Metrix during training
        #adj = torch.ones(features.shape[0], 25, 25).cuda()
        obj_num = features.shape[1]
        features_exp = features.unsqueeze(1)
        features_exp = features_exp.expand(-1, obj_num, -1, -1)
        features_exp_permute = features_exp.permute(0, 2, 1, 3)
        features_concat = torch.cat([features_exp, features_exp_permute], dim = -1)
        adj = self.Linear(features_concat).squeeze()
        features = features.squeeze()
        GCN_out_1 = self.GCN1(adj, features)
        relu_GCN_out_1 = F.relu(GCN_out_1)
        GCN_out_2 = self.GCN2(adj, relu_GCN_out_1)

        return GCN_out_2
        pass

    pass

def sample_negatives(pred_score, pos_indices, neg_indices, num_negatives=2,
                        use_hard_negatives=False, negative_pool_size=0, num_hard=2, drop_topk=0):
    """ Sample negatives from a set of indices. Several sampling strategies are supported:
    1, random; 2, hard negatives; 3, drop_topk hard negatives; 4, mix easy and hard negatives
    5, sampling within a pool of hard negatives; 6, sample across images of the same video.
    Args:
        pred_score: (num_img, num_words, num_region)
        pos_indices: (N_pos, 3) all positive region indices for the same word, not necessaryily the same image.
        neg_indices: (N_neg, 3) ...
        num_negatives (int):
        use_hard_negatives (bool):
        negative_pool_size (int):
        num_hard (int):
        drop_topk (int):
    Returns:

    """
    num_unique_pos = len(pos_indices)
    sampled_pos_indices = torch.cat([pos_indices] * num_negatives, dim=0)
    if use_hard_negatives:
        # print("using use_hard_negatives")
        neg_scores = pred_score[neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]  # TODO
        max_indices = torch.sort(neg_scores, descending=True)[1].tolist()
        if negative_pool_size > num_negatives:  # sample from a pool of hard negatives
            hard_pool = max_indices[drop_topk:drop_topk + negative_pool_size]
            hard_pool_indices = neg_indices[hard_pool]
            num_hard_negs = num_negatives
            sampled_easy_neg_indices = []
            if num_hard < num_negatives:
                easy_pool = max_indices[drop_topk + negative_pool_size:]
                easy_pool_indices = neg_indices[easy_pool]
                num_hard_negs = num_hard
                num_easy_negs = num_negatives - num_hard_negs
                sampled_easy_neg_indices = easy_pool_indices[
                    torch.randint(low=0, high=len(easy_pool_indices),
                                    size=(num_easy_negs * num_unique_pos, ), dtype=torch.long)
                ]
            sampled_hard_neg_indices = hard_pool_indices[
                torch.randint(low=0, high=len(hard_pool_indices),
                                size=(num_hard_negs * num_unique_pos, ), dtype=torch.long)
            ]

            if len(sampled_easy_neg_indices) != 0:
                sampled_neg_indices = torch.cat([sampled_hard_neg_indices, sampled_easy_neg_indices], dim=0)
            else:
                sampled_neg_indices = sampled_hard_neg_indices

        else:  # directly take the top negatives
            sampled_neg_indices = neg_indices[max_indices[drop_topk:drop_topk+len(sampled_pos_indices)]]
    else:
        sampled_neg_indices = neg_indices[
            torch.randint(low=0, high=len(neg_indices), size=(len(sampled_pos_indices),), dtype=torch.long)
        ]
    return sampled_pos_indices, sampled_neg_indices

def get_att_loss(opt, scores, att_labels, target, words, vid_names, qids, q_lens, img_indices, boxes,
                    start_indices, num_negatives=2, use_hard_negatives=False, drop_topk=0):
    """ compute ranking loss, use for loop to find the indices,
    use advanced indexing to perform the real calculation
    Build a list contains a quaduple

    Args:
        scores: cosine similarity scores (N, 5, Li, Lqa, Lr), in the range [-1, 1]
        att_labels: list(tensor), each has dimension (#num_imgs, #num_words, #regions), not batched
        target: 1D tensor (N, )
        words: LongTensor (N, 5, Lqa)
        vid_names: list(str) (N,)
        qids: list(int), (N, )
        q_lens: list(int), (N, )
        img_indices: list(list(int)), (N, Li), or None
        boxes: list(list(box)) of length N, each sublist represent an image,
            each box contains the coordinates of xyxy, or None
        num_negatives: number of negatives for each positive region
        use_hard_negatives: use hard negatives, uselect negatives with high scores
        drop_topk: drop topk highest negatives (since the top negatives might be correct, they are just not labeled)
        start_indices (list of int): each element is an index (at 0.5fps) of the first image
            with spatial annotation. If with_ts, set to zero
    Returns:
        att_loss: loss value for the batch
        att_predictions: (list) [{"gt": gt_scores, "pred": pred_scores}, ], used to calculate att. accuracy
    """
    pos_container = []  # contains tuples of 5 elements, which are (batch_i, ca_i, img_i, word_i, region_i)
    neg_container = []
    for batch_idx in range(len(target)):  # batch
        ca_idx = target[batch_idx].cpu().item()
        gt_score = att_labels[batch_idx]  # num_img * (num_words, num_region)
        start_idx = start_indices[batch_idx]  # int
        num_img = len(gt_score)
        sen_l, _ = gt_score[0].shape
        pred_score = scores[batch_idx, ca_idx, :num_img, :sen_l]  # (num_img, num_words, num_region)

        # find positive and negative indices
        batch_pos_indices = []
        batch_neg_indices = []
        for img_idx, img_gt_score in enumerate(gt_score):
            img_idx = start_idx + img_idx
            img_pos_indices = torch.nonzero(img_gt_score)  # (N_pos, 2) ==> (#words, #regions)
            if len(img_pos_indices) == 0:  # skip if no positive indices
                continue
            img_pos_indices = torch.cat([img_pos_indices.new_full([len(img_pos_indices), 1], img_idx),
                                            img_pos_indices], dim=1)  # (N_pos, 3) ==> (#img, #words, #regions)

            img_neg_indices = torch.nonzero(img_gt_score == 0)  # (N_neg, 2)
            img_neg_indices = torch.cat([img_neg_indices.new_full([len(img_neg_indices), 1], img_idx),
                                            img_neg_indices], dim=1)  # (N_neg, 3)

            batch_pos_indices.append(img_pos_indices)
            batch_neg_indices.append(img_neg_indices)

        if len(batch_pos_indices) == 0:  # skip if empty ==> no gt label for the video
            continue
        batch_pos_indices = torch.cat(batch_pos_indices, dim=0)  # (N_pos, 3) -->
        batch_neg_indices = torch.cat(batch_neg_indices, dim=0)  # (N_neg, 3)

        # sample positives and negatives
        available_img_indices = batch_pos_indices[:, 0].unique().tolist()
        for img_idx in available_img_indices:
            # pos_indices for a certrain img
            img_idx_pos_indices = batch_pos_indices[batch_pos_indices[:, 0] == img_idx]
            img_idx_neg_indices = batch_neg_indices[batch_neg_indices[:, 0] == img_idx]
            available_word_indices = img_idx_pos_indices[:, 1].unique().tolist()
            for word_idx in available_word_indices:
                # positives and negatives for a given image-word pair, specified by img_idx-word_idx
                img_idx_word_idx_pos_indices = img_idx_pos_indices[img_idx_pos_indices[:, 1] == word_idx]
                img_idx_word_idx_neg_indices = img_idx_neg_indices[img_idx_neg_indices[:, 1] == word_idx]
                # actually all the positives, not sampled pos
                sampled_pos_indices, sampled_neg_indices = \
                    sample_negatives(pred_score,
                                            img_idx_word_idx_pos_indices, img_idx_word_idx_neg_indices,
                                            num_negatives=num_negatives, use_hard_negatives=use_hard_negatives,
                                            negative_pool_size=opt.negative_pool_size,
                                            num_hard=opt.num_hard, drop_topk=drop_topk)

                base_indices = torch.LongTensor([[batch_idx, ca_idx]] * len(sampled_pos_indices)).\
                    to(sampled_pos_indices.device)
                pos_container.append(torch.cat([base_indices, sampled_pos_indices], dim=1))
                neg_container.append(torch.cat([base_indices, sampled_neg_indices], dim=1))

    pos_container = torch.cat(pos_container, dim=0)
    neg_container = torch.cat(neg_container, dim=0)

    # contain all the predictions and gt labels in this batch, only consider the ones with gt labels
    # also only consider the positive answer.
    att_predictions = None
    if not opt.vfeat_flag:
        att_predictions = dict(det_q=[],
                                det_ca=[])
        unique_pos_container = np.unique(pos_container.cpu().numpy(), axis=0)  # unique rows in the array
        for row in unique_pos_container:
            batch_idx, ca_idx, img_idx, word_idx, region_idx = row
            start_idx = start_indices[batch_idx]  # int
            cur_q_len = q_lens[batch_idx]
            num_region = att_labels[batch_idx][img_idx-start_idx].shape[1]  # num_img * (num_words, num_region)
            if len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()) != \
                    len(boxes[batch_idx][img_idx-start_idx]):
                print("scores[batch_idx, ca_idx, img_idx, word_idx].data.cpu()",
                        len(scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu()))
                print("len(boxes[batch_idx][img_idx-start_idx])", len(boxes[batch_idx][img_idx-start_idx]))
                print("boxes, batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx",
                        batch_idx, img_idx, start_idx, img_idx - start_idx, word_idx)
                print(row)
                raise AssertionError
            cur_det_data = {
                    "pred": scores[batch_idx, ca_idx, img_idx, word_idx, :num_region].data.cpu(),
                    "word": words[batch_idx, ca_idx, word_idx],
                    "qid": qids[batch_idx],
                    "vid_name": vid_names[batch_idx],
                    "img_idx": img_indices[batch_idx][img_idx],  # full indices
                    "boxes": boxes[batch_idx][img_idx-start_idx]  # located boxes
                }
            if word_idx < cur_q_len:
                att_predictions["det_q"].append(cur_det_data)
            else:
                att_predictions["det_ca"].append(cur_det_data)

    pos_scores = scores[pos_container[:, 0], pos_container[:, 1], pos_container[:, 2],
                        pos_container[:, 3], pos_container[:, 4]]
    neg_scores = scores[neg_container[:, 0], neg_container[:, 1], neg_container[:, 2],
                        neg_container[:, 3], neg_container[:, 4]]

    if opt.att_loss_type == "hinge":
        # max(0, m + S_pos - S_neg)
        att_loss = torch.clamp(opt.margin + neg_scores - pos_scores, min=0).sum()
    elif opt.att_loss_type == "lse":
        # log[1 + exp(scale * (S_pos - S_neg))]
        att_loss = torch.log1p(torch.exp(opt.alpha * (neg_scores - pos_scores))).sum()
    else:
        raise NotImplementedError("Only support hinge and lse")
    return att_loss, att_predictions

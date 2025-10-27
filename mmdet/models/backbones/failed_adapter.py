class ODConvAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(ODConvAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.norm = nn.GroupNorm(1, attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    """ kernel_size = 1 or 3 """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = ODConvAttention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

class ODMona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = ODConv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=4, reduction=0.0625, kernel_num=3)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class Simple_Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()


        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):

        x = self.norm(x) * self.gamma + x * self.gammax
        return x 

class ClassawareVisualSeekerAdapter(nn.Module):
    """
    类别感知的动态寻找稀疏token并进行增强
    每个query对应一个类别，只根据对应类别的原型进行更新
    """
    def __init__(self,
                 in_dim,
                 down_project_dim=64,  # 降维维度
                 num_classes=10,  # 类别数量，query数量将等于类别数量
                 k=64,  # topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 prototype_update_momentum=0.9,  # 原型更新动量
                 temperature=0.1,  # 相似度温度参数
                 roi_size=(3, 3),  # ROI Align的输出尺寸
                 ):
        super().__init__()
        
        self.num_classes = num_classes
        self.m = num_classes  # query数量等于类别数量
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.prototype_update_momentum = prototype_update_momentum
        self.temperature = temperature
        self.roi_size = roi_size

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        
        # Query/Prototype 初始化为 nn.Parameter，每个query对应一个类别
        self.m_queries = nn.Parameter(torch.randn(1, num_classes, down_project_dim))
        self.query_init = False
        
        # 类别初始化标志：记录每个类别是否已经初始化过
        self.class_init_flags = torch.zeros(num_classes, dtype=torch.bool)
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))
        # 原型库状态跟踪
        self.prototype_usage_count = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def initialize_queries_with_gt(self, all_gt_features, all_gt_labels, noise_std=0.1):
        """
        基于GT特征增量初始化query，每个query对应一个类别
        只初始化那些还没有初始化过且有GT样本的类别
        
        Args:
            all_gt_features: 所有GT特征 [total_gt, C_proj]
            all_gt_labels: 所有GT标签 [total_gt]
            noise_std: 噪声标准差
        """
        if len(all_gt_features) == 0:
            return
            
        # 计算所有GT特征的平均值和标准差
        #global_avg_feature = all_gt_features.mean(dim=0)  # [C_proj]
        global_std_feature = all_gt_features.std(dim=0)  # [C_proj]
        
        # 按类别分组计算平均特征
        unique_labels = torch.unique(all_gt_labels)
        class_avg_features = {}
        
        for label in unique_labels:
            label_mask = (all_gt_labels == label)
            label_features = all_gt_features[label_mask]
            if len(label_features) > 0:
                class_avg_features[label.item()] = label_features.mean(dim=0)
        
        # 找出需要初始化的类别（未初始化且有GT样本）
        classes_to_init = []
        for class_id in class_avg_features.keys():
            if not self.class_init_flags[class_id]:
                classes_to_init.append(class_id)
        
        if not classes_to_init:
            print("All classes have been initialized, no new initialization needed")
            return
        
        # 获取特征维度
        C_proj = self.m_queries.shape[2]  # 特征维度
        
        # 只初始化需要初始化的类别
        for class_id in classes_to_init:
            if class_id in class_avg_features:
                # 使用该类别的平均特征
                class_feature = class_avg_features[class_id]
                
                # 自适应噪声：基于特征的标准差调整噪声强度
                adaptive_noise_std = torch.clamp(global_std_feature.mean() * 0.5, min=0.1, max=0.5)
                actual_noise_std = max(noise_std, adaptive_noise_std.item())
                
                # 添加噪声以增加多样性
                noise = torch.randn_like(class_feature) * actual_noise_std
                initialized_feature = class_feature + noise
                
                # 更新对应类别的query
                with torch.no_grad():
                    self.m_queries.data[0, class_id] = initialized_feature
                
                # 标记该类已初始化
                self.class_init_flags[class_id] = True
                
                print(f"Class {class_id} initialized with {len(all_gt_features[all_gt_labels == class_id])} GT features, noise_std={actual_noise_std:.3f}")
        
        # 检查是否所有类别都已初始化
        if self.class_init_flags.all():
            self.query_init = True
            print(f"All {self.num_classes} classes have been initialized")
        else:
            remaining_classes = (~self.class_init_flags).sum().item()
            print(f"Initialized {len(classes_to_init)} new classes, {remaining_classes} classes remaining")

    def extract_gt_features(self, activated_features, hw_shapes, gt_info):
        """
        从激活后的特征中提取GT区域特征
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
            gt_info: GT信息字典，包含bboxes, labels等
            hw_shape: 特征图尺寸 (H, W)
        
        Returns:
            gt_features: GT特征列表 [B个元素，每个是[num_gt, C_proj]]
            gt_labels: GT标签列表 [B个元素，每个是[num_gt]]
        """
        B, N, C_proj = activated_features.shape
        H, W = hw_shapes
        
        # 获取GT在特征图上的位置
        feature_positions = get_feature_positions(gt_info, hw_shapes)
        
        gt_features = []
        gt_labels = []
        
        for batch_idx in range(B):
            batch_features = activated_features[batch_idx]  # [N, C_proj]
            batch_positions = feature_positions[batch_idx]  # [num_gt, 4]
            batch_labels = gt_info['labels'][batch_idx]  # [num_gt]
            
            if len(batch_positions) == 0:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 将特征重塑为空间形式 [H, W, C_proj]
            spatial_features = batch_features.view(H, W, C_proj).permute(2, 0, 1)  # [C_proj, H, W]
            
            # 准备ROI Align的输入
            rois = []
            valid_gt_indices = []
            
            for gt_idx, bbox in enumerate(batch_positions):
                x1, y1, x2, y2 = bbox.float()
                
                # 确保坐标在有效范围内
                x1 = torch.clamp(x1, 0, W)
                y1 = torch.clamp(y1, 0, H)
                x2 = torch.clamp(x2, 0, W)
                y2 = torch.clamp(y2, 0, H)
                
                # 检查bbox是否有效
                if x2 <= x1 or y2 <= y1:
                    continue
                # ROI Align期望的格式：[batch_idx, x1, y1, x2, y2]
                roi = torch.tensor([0, x1, y1, x2, y2], dtype=torch.float32, device=spatial_features.device)
                rois.append(roi)
                valid_gt_indices.append(gt_idx)
            
            if not rois:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 转换为tensor
            rois = torch.stack(rois)  # [num_valid_gt, 5]
            valid_labels = batch_labels[valid_gt_indices]
            
            # 使用ROI Align提取特征
            spatial_features = spatial_features.unsqueeze(0)  # [1, C_proj, H, W]
            
            try:
                roi_features = roi_align(
                    spatial_features,  # [1, C_proj, H, W]
                    rois,             # [num_valid_gt, 4] - [x1, y1, x2, y2]
                    output_size=self.roi_size,  # (3, 3)
                    spatial_scale=1.0,  # 需要根据实际特征图尺寸调整
                    sampling_ratio=-1,
                    aligned=True
                )  # [num_valid_gt, C_proj, roi_h, roi_w]
                
                # 全局平均池化得到特征向量
                roi_features = roi_features.mean(dim=(2, 3))  # [num_valid_gt, C_proj]
                
                gt_features.append(roi_features)
                gt_labels.append(valid_labels)
                
            except Exception as e:
                # 如果ROI Align失败，使用中心点特征作为备选方案
                print(f"ROI Align failed: {e}, using center point features")
                center_features = []
                for bbox in batch_positions[valid_gt_indices]:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 使用clamp确保中心点坐标在有效范围内
                    center_x = torch.clamp(center_x, 0, W)
                    center_y = torch.clamp(center_y, 0, H)
                
                    center_feature = batch_features[center_y * W + center_x]  # [C_proj]
                    center_features.append(center_feature)
                
                if center_features:
                    center_features = torch.stack(center_features)  # [num_valid_gt, C_proj]
                    gt_features.append(center_features)
                    gt_labels.append(valid_labels)
                else:
                    gt_features.append(torch.empty(0, C_proj))
                    gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
        
        return gt_features, gt_labels

    def update_prototypes_with_gt(self, activated_features, hw_shapes, gt_info):
        """
        基于GT信息更新query/prototype，每个query只更新对应类别的原型
        """
        if gt_info is None:
            return
            
        gt_features, gt_labels = self.extract_gt_features(activated_features, hw_shapes, gt_info)
        
        # 更清晰的empty情况处理
        if not gt_features:
            return
        
        # 过滤掉empty的特征和标签
        valid_features = []
        valid_labels = []
        
        for features, labels in zip(gt_features, gt_labels):
            if len(features) > 0 and len(labels) > 0:
                valid_features.append(features)
                valid_labels.append(labels)
        
        # 如果没有有效的GT特征，直接返回
        if not valid_features:
            return
            
        # 拼接所有有效的GT特征
        all_gt_features = torch.cat(valid_features, dim=0)  # [total_valid_gt, C_proj]
        all_gt_labels = torch.cat(valid_labels, dim=0)  # [total_valid_gt]
        
        # 如果query还没有初始化，先进行初始化
        if not self.query_init:
            self.initialize_queries_with_gt(all_gt_features, all_gt_labels)
            return  # 第一次调用只进行初始化，不进行更新
        
        # 按类别分组更新原型，每个query只更新对应类别的原型
        unique_labels = torch.unique(all_gt_labels)
        
        for label in unique_labels:
            label_mask = (all_gt_labels == label)
            label_features = all_gt_features[label_mask]  # [num_gt_class, C_proj]
            
            if len(label_features) == 0:
                continue
                
            # 计算该类别的平均特征
            avg_gt_feature = label_features.mean(dim=0)  # [C_proj]
            
            # 直接更新对应类别的query（类别ID作为query索引）
            class_id = label.item()
            if class_id < self.num_classes:  # 确保类别ID在有效范围内
                with torch.no_grad():
                    # 使用动量更新对应类别的query
                    self.m_queries.data[0, class_id] = (
                        self.prototype_update_momentum * self.m_queries.data[0, class_id] +
                        (1 - self.prototype_update_momentum) * avg_gt_feature
                    )
                    # 更新使用计数
                    self.prototype_usage_count[class_id] += 1

    def select_tokens_with_prototypes(self, activated_features):
        """
        基于原型特征选择最相关的tokens
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
        
        Returns:
            topk_indices: topk索引 [B, k]
            sparse_image_tokens: 选中的稀疏tokens [B, k, C_proj]
        """
        B, N, C_proj = activated_features.shape
        
        # 计算图像特征与原型query的相似度
        norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # [B, N, C_proj]
        norm_queries = F.normalize(self.m_queries.expand(B, -1, -1), p=2, dim=-1)  # [B, num_classes, C_proj]
        
        # 计算相似度矩阵
        similarities = torch.bmm(norm_image_features, norm_queries.transpose(1, 2))  # [B, N, num_classes]
        
        # 对每个类别query，选择最相似的tokens
        all_scores = []
        for b in range(B):
            batch_similarities = similarities[b]  # [N, num_classes]
            # 取每个类别query对应的最大相似度
            max_similarities, _ = batch_similarities.max(dim=1)  # [N]
            all_scores.append(max_similarities)
        
        all_scores = torch.stack(all_scores)  # [B, N]
        
        # 使用STE生成topk mask
        mask = ste_topk_mask(all_scores, self.k)  # [B, N]
        
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:, :, 1]  # [B, k]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        
        # 提取稀疏tokens
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)  # [B, k, C_proj]
        
        return topk_indices, sparse_image_tokens, topk_indices_expanded

    def forward(self, image_features, hw_shapes=None, gt_info=None):
        """
        Args:
            image_features: 输入特征 [B, N, C]
            hw_shapes: 特征图尺寸列表
            gt_info: GT信息字典，包含gt_features等
        """
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 降维和激活 ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features))  # [B, N, C_proj]

        # --- 2. 基于GT信息更新原型 ---
        if gt_info is not None and hw_shapes is not None:
            self.update_prototypes_with_gt(activated_features, hw_shapes, gt_info)

        # --- 3. 基于原型选择tokens ---
        topk_indices, sparse_image_tokens, topk_indices_expanded = self.select_tokens_with_prototypes(activated_features)

        # --- 4. 原型与稀疏tokens的交互 ---
        queries = self.m_queries.expand(B, -1, -1)  # [B, num_classes, C_proj]
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)  # [B, num_classes+k, C_proj]
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的稀疏tokens（不更新query）
        enhanced_sparse_tokens = enhanced_sequence[:, self.num_classes:, :]  # [B, k, C_proj]

        # --- 5. 信息还原与升维 ---
        # 创建零张量并填充增强后的tokens
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 升维
        delta_x = self.up_project(delta_x_projected)  # [B, N, C_in]
        
        return identity + delta_x * self.gamma

class ESODQuerySeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强,引入esod部分使用gt来增强模型的感知
    """
    def __init__(self,
                 in_dim,
                 text_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.text_dim = text_dim
        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        # --- 结构结束 ---
        # Query 初始化为 nn.Parameter，起到码表的作用
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        self.text_projector = nn.Linear(text_dim,down_project_dim)
        self.text_nonlinear_activation = F.gelu
        self.text_reconstructor = nn.Linear(down_project_dim, text_dim)
        self.image_reconstructor = nn.Linear(down_project_dim, down_project_dim)
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(8e-1))
        self.dwconv = DWConvForObjSeeker(down_project_dim, down_project_dim, kernel_size=13, stride=1)
        self.heatmap_conv = Segmenter(nc=1,ch=down_project_dim)


    def forward(self, image_features, hw_shapes=None, text_features=None,gt_info=None):
        #1.常规adapter操作：对视觉特征进行降维
        #2.文本量化投影：将文本投影降维，并通过欧式距离量化到query上
        #3.辅助损失1：query尝试重建到文本原特征，L2Norm计算损失，保证query和文本的关联
        #4.辅助损失2：query尝试重建到视觉池化特征上，同样L2Norm计算损失，保证query和视觉的关联
        #5.辅助损失3：query和文本本身投影特征，L2Norm计算损失，保证query和文本投影的更新
        #6.辅助损失4：预测esod以保证
        #6.基于query寻找相似度最高topk的图像特征块
        #7.稀疏attn增强topk特征
        #8.升维+残差
        assert text_features is not None, "text_features 不能为 None"
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 对输入特征应用 Adapter 前半部分：降维 -> GELU -> Dropout ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features)) # (B, N, C_proj)

        b, n, c = activated_features.shape
        h, w = hw_shapes
        pred = activated_features.reshape(b, h, w, c).permute(0, 3, 1, 2)
        pred = self.dwconv(pred)
        pred = self.heatmap_conv(pred)
        if gt_info is not None:
            mask,weight = gen_adapter_mask(gt_info,hw_shapes)
            loss_seg = compute_loss_seg(pred,mask,weight)

        # --- 2. 文本量化投影：将文本投影降维，并通过欧式距离量化到query上 ---
        # 使用文本投影层将文本特征投影到降维空间
        text_features = text_features.detach()
        projected_text_feat = self.text_projector(text_features)  # (B, L, C_proj)
        projected_text_feat = self.text_nonlinear_activation(projected_text_feat)
        
        # 使用全部文本token进行计算
        L_query = self.m
        L_text = projected_text_feat.shape[1]
        queries = self.m_queries.expand(B, -1, -1)  # (B, m, C_proj)
        norm_queries = F.normalize(queries, p=2, dim=-1)
        norm_text_feat = F.normalize(projected_text_feat, p=2, dim=-1)
        
        # 计算query与文本的相似度
        query_text_similarity = torch.bmm(norm_queries, norm_text_feat.transpose(1, 2))  # (B, m, L)
        
        # 选择最相似的query进行量化 (每个文本token对应一个最相似的query)
        quantized_query_idx = query_text_similarity.argmax(dim=1)  # (B, L)
        #quantized_queries = queries[torch.arange(B).unsqueeze(1), quantized_query_idx]  # (B, L, C_proj)
        quantized_queries = torch.gather(
            queries, dim=1,
            index=quantized_query_idx.unsqueeze(-1).expand(-1, -1, queries.size(-1))
        )  # (B, L, C_proj)
        # --- 3-5. 计算辅助损失 ---
        # 辅助损失1：query尝试重建到文本原特征 (使用全部文本token)
        text_reconstructed = self.text_reconstructor(quantized_queries)  # (B, L, text_dim)
        loss1 = 0.1 * F.mse_loss(text_reconstructed, text_features)
        
        # 辅助损失2：query尝试重建到视觉池化特征
        image_pooled = activated_features.mean(dim=1, keepdim=True)  # (B, 1, C_proj)
        image_pooled_expanded = image_pooled.expand(-1, L_text, -1)  # (B, L, C_proj)
        image_reconstructed = self.image_reconstructor(quantized_queries)  # (B, L, C_proj)
        loss2 = 0.1 * F.mse_loss(image_reconstructed, image_pooled_expanded)
        
        # 辅助损失3：像VQVAE一样，让query特征和文本降维后的特征分别梯度解耦然后算损失
        # 对quantized_queries和projected_text_feat分别进行梯度截断
        quantized_queries_detached = quantized_queries.detach()
        projected_text_feat_detached_for_loss = projected_text_feat.detach()
        loss3 = 0.1 *F.mse_loss(quantized_queries_detached, projected_text_feat)+0.25*0.1*F.mse_loss(quantized_queries, projected_text_feat_detached_for_loss)
        
        # 总辅助损失
        #aux_loss = (loss1 + loss3)*5
        aux_loss = (loss1 + loss2 + loss3)*2
        if gt_info is not None:
            # print("loss_seg is")
            # print(loss_seg)
            # print("*"*100)
            aux_loss = aux_loss + loss_seg.mean()*2

        # --- 6. 基于query寻找相似度最高topk的图像特征块 ---
        # 使用量化后的query计算与图像特征的相似度
        norm_quantized_queries = F.normalize(quantized_queries, p=2, dim=-1)  # (B, L, C_proj)
        norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # (B, N, C_proj)
        image_scores = torch.bmm(norm_image_features, norm_quantized_queries.transpose(1, 2))  # (B, N, L)
        # 在文本token维度取最大值，得到每个图像token与所有文本token的最大相似度
        image_scores, _ = image_scores.max(dim=2)  # (B, N)
        
        # 使用STE生成topk mask
        mask = ste_topk_mask(image_scores, self.k)  # (B, N)
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:,:,1]  # (B, k)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)

        # --- 7. 稀疏attn增强topk特征 ---
        # 将量化后的query与稀疏图像token结合
        combined_sequence = torch.cat([quantized_queries, sparse_image_tokens], dim=1)  # (B, L+k, C_proj)
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的图像token
        enhanced_sparse_tokens = enhanced_sequence[:, L_text:, :]  # (B, k, C_proj)

        # --- 8. 升维+残差 ---
        # 创建一个与激活后特征图形状相同的零张量
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 应用 Adapter 后半部分：升维
        delta_x = self.up_project(delta_x_projected)  # -> (B, N, C_in)
        
        return identity + delta_x * self.gamma, aux_loss

    def old_forward(self, image_features, hw_shapes=None, text_features=None):
        #1.常规adapter操作：对视觉特征进行降维
        #2.文本投影：将文本投影降维或者截断，降维则产生和视觉池化特征匹配的损失
        #3.文本寻找相似度最高topk
        #4.可学习query稀疏attn增强topk特征，ema更新query
        #5.升维+残差
        # return identity + delta_x * self.gamma
        return image_features

class OmniDynamicSeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强，区别是将文本和视觉对齐到了同一个空间
    """
    def __init__(self,
                 in_dim,
                 text_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim,text_dim)
        self.omni_down_project = nn.Linear(text_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        # --- 结构结束 ---
        
        # Query 初始化为 nn.Parameter
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))



    def forward(self, image_features, hw_shapes=None,text_features=None):
        assert text_features is not None, "text_features 不能为 None"
        #TODO：1.实现辅助loss 2.增加修改输出为维度（）
        B, N, C_in = image_features.shape
        identity = image_features
        
        # --- 1. 对输入特征应用 Adapter 前半部分：降维 -> GELU -> Dropout ---
        projected_features = self.down_project(image_features) #到text_dim了
        projected_features = self.dropout(self.nonlinear_activation(projected_features))
        activated_features = self.omni_down_project(projected_features) # -> (B, N, C_proj)

        # --- 2. 文本引导与 Top-k 选择 (在激活后的特征上进行) ---
        # #原来的
        pooled_text_feat = text_features[:,0] #使用CLS token
        projected_text_feat = self.omni_down_project(pooled_text_feat).unsqueeze(1)
        eps = 1e-8
        norm_image_features = F.normalize(activated_features+eps, p=2, dim=-1)
        norm_key_text_tokens = F.normalize(projected_text_feat+eps, p=2, dim=-1)
        image_scores = torch.bmm(norm_image_features, norm_key_text_tokens.transpose(1, 2)).squeeze(-1)

        
        # --------- 用STE生成mask ---------
        mask = ste_topk_mask(image_scores, self.k)  # (B, N)
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:,:,1]  # (B, k)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)

        # --- 3. 简化版注意力交互 ---
        queries = self.m_queries.expand(B, -1, -1)
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        # 只有一个残差连接，没有后续 FFN
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的图像 token和query
        enhanced_queries = enhanced_sequence[:, :self.m, :]  # (B, m, C_proj)
        enhanced_sparse_tokens = enhanced_sequence[:, self.m:, :]

        # 替换原本的 query 参数，使用不带梯度的EMA平滑更新
        with torch.no_grad():
            self.m_queries.copy_(0.8 * self.m_queries + 0.2 * enhanced_queries.mean(dim=0, keepdim=True))
        
        # --- 4. 信息还原与升维 ---
        # 创建一个与激活后特征图形状相同的零张量
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 应用 Adapter 后半部分：升维
        delta_x = self.up_project(delta_x_projected) # -> (B, N, C_in)
        
        return identity + delta_x * self.gamma

class HyperAdapterMona(BaseModule):
    def __init__(self,
                 in_dim,
                 m=16):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        # 不将generator作为模块参数存储，而是通过外部传入
        self.m_queries = nn.Parameter(torch.randn(1, m, 64))
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_conv = None
        self.static_conv = MonaOp(64)

    def forward(self, x, generator, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 计算 z：基于 m_queries 与 project1 的相似度得到注意力分数，对所有特征加权平均
        # project1: [b, c, h, w] -> [b, hw, c]
        x_flat = project1.reshape(b, c, h * w).permute(0, 2, 1)
        # m_queries: [1, m, c] -> [b, m, c]
        q = self.m_queries.expand(b, -1, -1)
        # 相似度: [b, m, c] x [b, c, hw] -> [b, m, hw]
        sim = torch.matmul(q, x_flat.transpose(1, 2))
        attn = F.softmax(sim, dim=-1)
        # 加权求和: [b, m, hw] x [b, hw, c] -> [b, m, c]
        z_per_query = torch.matmul(attn, x_flat)
        # 对 m 维求平均，得到 [b, c]
        z = z_per_query.mean(dim=1)
        
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if self.hyper_conv is None:
            self.hyper_conv = HyperConv2d(padding=generator.f_size//2)
        old_project1 = project1
        project1 = self.hyper_conv(project1, z, generator)
        old_project1 = self.static_conv(old_project1)
        project1 = project1 + old_project1
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

#TODO:设计多路的HyperAdapter
class HyperAdapterMulti(BaseModule):
    def __init__(self,
                 in_dim,
                 m=16):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        # 不将generator作为模块参数存储，而是通过外部传入
        self.m_queries = nn.Parameter(torch.randn(1, m, 64))
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_convs = []

    def forward(self, x, generators, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 计算 z：基于 m_queries 与 project1 的相似度得到注意力分数，对所有特征加权平均
        # project1: [b, c, h, w] -> [b, hw, c]
        x_flat = project1.reshape(b, c, h * w).permute(0, 2, 1)
        # m_queries: [1, m, c] -> [b, m, c]
        q = self.m_queries.expand(b, -1, -1)
        # 相似度: [b, m, c] x [b, c, hw] -> [b, m, hw]
        sim = torch.matmul(q, x_flat.transpose(1, 2))
        attn = F.softmax(sim, dim=-1)
        # 加权求和: [b, m, hw] x [b, hw, c] -> [b, m, c]
        z_per_query = torch.matmul(attn, x_flat)
        # 对 m 维求平均，得到 [b, c]
        z = z_per_query.mean(dim=1)
        
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if len(self.hyper_convs) == 0:
            for generator in generators:
                self.hyper_convs.append(HyperConv2d(padding=generator.f_size//2))

        # 自动同步设备（避免隐式跨设备拷贝）
        x_device = x.device
        for gen in generators:
            try:
                gen_device = next(gen.parameters()).device
                if gen_device != x_device:
                    gen.to(x_device)
            except StopIteration:
                # 如果 generator 没有参数，跳过设备检查
                pass

        # 流式融合以降低显存峰值和分支复杂度
        if len(generators) > 0:
            fused_sum = None
            for i, generator in enumerate(generators):
                out_i = self.hyper_convs[i](project1, z, generator)
                if fused_sum is None:
                    fused_sum = out_i
                else:
                    fused_sum = fused_sum + out_i
            fused = fused_sum / float(len(generators))
            project1 = project1 + fused
        
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

#TODO:视觉-文本两路特征卷积
class HyperAdapterVL(BaseModule):
    def __init__(self,
                 in_dim,
                 ):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        # 不将generator作为模块参数存储，而是通过外部传入
        
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.gamma_vis = nn.Parameter(torch.tensor(5e-1))
        
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_conv_vis = None
        self.hyper_conv_text = None

    def forward(self, x, generator, hw_shapes=None, text_features=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        z_vis = project1.mean(dim=1)
        z_text = text_features.mean(dim=1)[:,:64]
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 计算 z：基于 m_queries 与 project1 的相似度得到注意力分数，对所有特征加权平均
        # project1: [b, c, h, w] -> [b, hw, c]
        
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if self.hyper_conv_vis is None:
            self.hyper_conv_vis = HyperConv2d(padding=generator.f_size//2)
        if self.hyper_conv_text is None:
            self.hyper_conv_text = HyperConv2d(padding=generator.f_size//2)
        
        project1_vis = self.hyper_conv_vis(project1, z_vis, generator)
        project1_text = self.hyper_conv_text(project1, z_text, generator)
        project1 = project1_vis * self.gamma_vis + project1_text * (1 - self.gamma_vis)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
        

#TODO:利用esod寻找
class HyperAdapterESODSeeker(BaseModule):
    def __init__(self,
                 in_dim,
                 ):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        # 不将generator作为模块参数存储，而是通过外部传入
        
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.dwconv = DWConvForObjSeeker(64, 64, kernel_size=13, stride=1)
        self.heatmap_conv = Segmenter(nc=1,ch=64)
        
        # 预创建HyperConv2d，避免每次forward都创建新实例
        self.hyper_conv = None
        

    def forward(self, x, generator, hw_shapes=None, gt_info=None):
        if gt_info is not None:
            mask,weight = gen_adapter_mask(gt_info,hw_shapes)
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes

       
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 计算 z：基于 m_queries 与 project1 的相似度得到注意力分数，对所有特征加权平均
        # project1: [b, c, h, w] -> [b, hw, c]
        pred = self.dwconv(project1)
        pred = self.heatmap_conv(pred)
        # pred: [B, 1, H, W]，用 sigmo id（或直接二值）作为权重，不做 softmax
        weights = (pred > 0.2).float()  # 若需严格二值，可改为 (pred > 0).float()
        # 归一化加权平均，避免权重和为 0 的数值问题
        num = (project1 * weights).sum(dim=(2, 3))            # [B, C]
        den = weights.sum(dim=(2, 3)).clamp_min(1e-6)         # [B, 1]
        z = num / den                                         # [B, C]
        
        # 使用预创建的HyperConv2d，避免每次forward都创建新实例
        if self.hyper_conv is None:
            self.hyper_conv = HyperConv2d(padding=generator.f_size//2)
        project1 = self.hyper_conv(project1, z, generator)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)


        if gt_info is not None:
            loss_seg = compute_loss_seg(pred,mask,weight)
            return identity + project2,loss_seg*0.1
        else:
            return identity + project2

#TODO:第四处注入，需要形成原型库，使用原型库进行增强
class VisualSeekerAdapter(nn.Module):
    """
    动态寻找稀疏token并进行增强
    """
    def __init__(self,
                 in_dim,
                 down_project_dim=64,#降维维度
                 m=16,#m个query
                 k=64,#topk个token
                 attention_heads=4,
                 dropout_rate=0.1,
                 prototype_update_momentum=0.9,  # 原型更新动量
                 temperature=0.1,  # 相似度温度参数
                 roi_size=(3, 3),  # ROI Align的输出尺寸
                 ):
        super().__init__()
        
        self.m = m
        self.k = k
        self.in_dim = in_dim
        self.projected_dim = down_project_dim
        self.prototype_update_momentum = prototype_update_momentum
        self.temperature = temperature
        self.roi_size = roi_size

        # --- 核心 Adapter 结构 ---
        # 1. 降维
        self.down_project = nn.Linear(in_dim, down_project_dim)
        # 2. 非线性激活与 Dropout (紧跟降维之后)
        self.nonlinear_activation = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        # 3. 升维
        self.up_project = nn.Linear(down_project_dim, in_dim)
        
        #Query/Prototype 初始化为 nn.Parameter
        self.m_queries = nn.Parameter(torch.randn(1, m, down_project_dim))
        self.query_init = False
        # 原型匹配网络：将GT特征投影到query空间
        # self.prototype_matcher = nn.Sequential(
        #     nn.Linear(down_project_dim, down_project_dim),
        #     nn.ReLU(),
        #     nn.Linear(down_project_dim, down_project_dim)
        # )
        # ROI Align层，用于提取GT区域特征
        # self.roi_align = roi_align(
        #     output_size=roi_size,
        #     spatial_scale=1.0,  # 需要根据实际特征图尺寸调整
        #     sampling_ratio=-1,
        #     aligned=True
        # )
        
        # 交互模块: 只有一个自注意力层和 LayerNorm
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=down_project_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(down_project_dim)
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.tensor(1e-1))
        # 原型库状态跟踪
        self.prototype_usage_count = nn.Parameter(torch.zeros(m), requires_grad=False)

    def initialize_queries_with_gt(self, all_gt_features, all_gt_labels, noise_std=0.25):
        """
        基于GT特征的平均值初始化query，并添加噪声以增加多样性
        
        Args:
            all_gt_features: 所有GT特征 [total_gt, C_proj]
            all_gt_labels: 所有GT标签 [total_gt]
            noise_std: 噪声标准差
        """
        if len(all_gt_features) == 0:
            return
            
        # 计算所有GT特征的平均值和标准差
        global_avg_feature = all_gt_features.mean(dim=0)  # [C_proj]
        global_std_feature = all_gt_features.std(dim=0)  # [C_proj]
        
        # 按类别分组计算平均特征
        unique_labels = torch.unique(all_gt_labels)
        class_avg_features = {}
        
        for label in unique_labels:
            label_mask = (all_gt_labels == label)
            label_features = all_gt_features[label_mask]
            if len(label_features) > 0:
                class_avg_features[label.item()] = label_features.mean(dim=0)
        
        # 初始化query
        m = self.m_queries.shape[1]  # query数量
        C_proj = self.m_queries.shape[2]  # 特征维度
        
        # 策略1: 使用全局平均特征作为基础
        base_feature = global_avg_feature
        
        # 策略2: 如果有足够的类别，使用类别平均特征
        if len(class_avg_features) >= m:
            # 选择前m个类别的平均特征
            class_features = list(class_avg_features.values())[:m]
            base_features = torch.stack(class_features)  # [m, C_proj]
        else:
            # 使用全局平均特征，并添加不同方向的噪声
            base_features = base_feature.unsqueeze(0).expand(m, -1)  # [m, C_proj]
        
        # 自适应噪声：基于特征的标准差调整噪声强度
        # 如果特征变化很大，使用更大的噪声；如果特征变化很小，使用较小的噪声
        adaptive_noise_std = torch.clamp(global_std_feature.mean() * 0.5, min=0.1, max=0.5)
        actual_noise_std = max(noise_std, adaptive_noise_std.item())
        
        # 添加噪声以增加多样性
        noise = torch.randn_like(base_features) * actual_noise_std
        initialized_queries = base_features + noise
        
        # 更新query参数
        with torch.no_grad():
            self.m_queries.data.copy_(initialized_queries.unsqueeze(0))  # [1, m, C_proj]
        
        # 标记为已初始化
        self.query_init = True
        
        print(f"Query initialized with {len(unique_labels)} classes, {len(all_gt_features)} GT features, noise_std={actual_noise_std:.3f}")

    def extract_gt_features(self, activated_features, hw_shapes,gt_info):
        """
        从激活后的特征中提取GT区域特征
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
            gt_info: GT信息字典，包含bboxes, labels等
            hw_shape: 特征图尺寸 (H, W)
        
        Returns:
            gt_features: GT特征列表 [B个元素，每个是[num_gt, C_proj]]
            gt_labels: GT标签列表 [B个元素，每个是[num_gt]]
        """
        B, N, C_proj = activated_features.shape
        H, W = hw_shapes
        
        # 获取GT在特征图上的位置
        feature_positions = get_feature_positions(gt_info, hw_shapes)
        
        gt_features = []
        gt_labels = []
        
        for batch_idx in range(B):
            batch_features = activated_features[batch_idx]  # [N, C_proj]
            batch_positions = feature_positions[batch_idx]  # [num_gt, 4]
            batch_labels = gt_info['labels'][batch_idx]  # [num_gt]
            
            if len(batch_positions) == 0:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 将特征重塑为空间形式 [H, W, C_proj]
            spatial_features = batch_features.view(H, W, C_proj).permute(2, 0, 1)  # [C_proj, H, W]
            
            # 准备ROI Align的输入
            # 需要将特征图坐标转换为ROI Align期望的格式
            rois = []
            valid_gt_indices = []
            
            for gt_idx, bbox in enumerate(batch_positions):
                x1, y1, x2, y2 = bbox.float()
                
                # 确保坐标在有效范围内
                x1 = torch.clamp(x1, 0, W)
                y1 = torch.clamp(y1, 0, H)
                x2 = torch.clamp(x2, 0, W)
                y2 = torch.clamp(y2, 0, H)
                
                # 检查bbox是否有效
                if x2 <= x1 or y2 <= y1:
                    continue
                 # ROI Align期望的格式：[batch_idx, x1, y1, x2, y2]
                roi = torch.tensor([0, x1, y1, x2, y2], dtype=torch.float32, device=spatial_features.device)
                rois.append(roi)
                valid_gt_indices.append(gt_idx)
            if not rois:
                gt_features.append(torch.empty(0, C_proj))
                gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
                continue
            
            # 转换为tensor
            rois = torch.stack(rois)  # [num_valid_gt, 5]
            valid_labels = batch_labels[valid_gt_indices]
            
            # 使用ROI Align提取特征
            # 注意：ROI Align期望输入是[B, C, H, W]格式
            spatial_features = spatial_features.unsqueeze(0)  # [1, C_proj, H, W]
            
            try:
                roi_features = roi_align(
                    spatial_features,  # [1, C_proj, H, W]
                    rois,             # [num_valid_gt, 4] - [x1, y1, x2, y2]
                    output_size=self.roi_size,  # (3, 3)
                    spatial_scale=1.0,  # 需要根据实际特征图尺寸调整
                    sampling_ratio=-1,
                    aligned=True
                )  # [num_valid_gt, C_proj, roi_h, roi_w]
                
                # 全局平均池化得到特征向量
                roi_features = roi_features.mean(dim=(2, 3))  # [num_valid_gt, C_proj]
                
                gt_features.append(roi_features)
                gt_labels.append(valid_labels)
                
            except Exception as e:
                # 如果ROI Align失败，使用中心点特征作为备选方案
                print(f"ROI Align failed: {e}, using center point features")
                center_features = []
                for bbox in batch_positions[valid_gt_indices]:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # 使用clamp确保中心点坐标在有效范围内
                    center_x = torch.clamp(center_x, 0, W)
                    center_y = torch.clamp(center_y, 0, H)
                
                    center_feature = batch_features[center_y * W + center_x]  # [C_proj]
                    center_features.append(center_feature)
                
                if center_features:
                    center_features = torch.stack(center_features)  # [num_valid_gt, C_proj]
                    gt_features.append(center_features)
                    gt_labels.append(valid_labels)
                else:
                    gt_features.append(torch.empty(0, C_proj))
                    gt_labels.append(torch.empty(0, dtype=batch_labels.dtype))
        
        return gt_features, gt_labels

    def update_prototypes_with_gt(self, activated_features, hw_shapes, gt_info):
        """
        基于GT信息更新query/prototype
        """
        if gt_info is None:
            return
            
        gt_features, gt_labels = self.extract_gt_features(activated_features, hw_shapes, gt_info)
        
        # 更清晰的empty情况处理
        if not gt_features:
            return
        
        # 过滤掉empty的特征和标签
        valid_features = []
        valid_labels = []
        
        for features, labels in zip(gt_features, gt_labels):
            if len(features) > 0 and len(labels) > 0:
                valid_features.append(features)
                valid_labels.append(labels)
        
        # 如果没有有效的GT特征，直接返回
        if not valid_features:
            return
            
        # 拼接所有有效的GT特征
        all_gt_features = torch.cat(valid_features, dim=0)  # [total_valid_gt, C_proj]
        all_gt_labels = torch.cat(valid_labels, dim=0)  # [total_valid_gt]
        
        # 如果query还没有初始化，先进行初始化
        if not self.query_init:
            self.initialize_queries_with_gt(all_gt_features, all_gt_labels)
            return  # 第一次调用只进行初始化，不进行更新
        
        # 按类别分组更新原型
        unique_labels = torch.unique(all_gt_labels)
        
        for label in unique_labels:
            label_mask = (all_gt_labels == label)
            label_features = all_gt_features[label_mask]  # [num_gt_class, C_proj]
            
            if len(label_features) == 0:
                continue
                
            # 计算该类别的平均特征
            avg_gt_feature = label_features.mean(dim=0)  # [C_proj]
            
            # 计算与现有原型的相似度
            current_prototypes = self.m_queries.data.squeeze(0)  # [m, C_proj]
            similarities = F.cosine_similarity(
                avg_gt_feature.unsqueeze(0), current_prototypes, dim=-1
            )  # [m]
            
            # 找到最相似的原型
            most_similar_idx = similarities.argmax()
            similarity_score = similarities[most_similar_idx]
            
            # 如果相似度足够高，更新该原型
            if similarity_score > 0.5:  # 相似度阈值
                with torch.no_grad():
                    # 使用动量更新
                    self.m_queries.data[0, most_similar_idx] = (
                        self.prototype_update_momentum * self.m_queries.data[0, most_similar_idx] +
                        (1 - self.prototype_update_momentum) * avg_gt_feature
                    )
                    # 更新使用计数
                    self.prototype_usage_count[most_similar_idx] += 1

    def select_tokens_with_prototypes(self, activated_features):
        """
        基于原型特征选择最相关的tokens
        
        Args:
            activated_features: 激活后的特征 [B, N, C_proj]
        
        Returns:
            topk_indices: topk索引 [B, k]
            sparse_image_tokens: 选中的稀疏tokens [B, k, C_proj]
        """
        B, N, C_proj = activated_features.shape
        
        # 计算图像特征与原型query的相似度
        norm_image_features = F.normalize(activated_features, p=2, dim=-1)  # [B, N, C_proj]
        norm_queries = F.normalize(self.m_queries.expand(B, -1, -1), p=2, dim=-1)  # [B, m, C_proj]
        
        # 计算相似度矩阵
        similarities = torch.bmm(norm_image_features, norm_queries.transpose(1, 2))  # [B, N, m]
        
        # 对每个原型，选择最相似的tokens
        all_scores = []
        for b in range(B):
            batch_similarities = similarities[b]  # [N, m]
            # 取每个原型对应的最大相似度
            max_similarities, _ = batch_similarities.max(dim=1)  # [N]
            all_scores.append(max_similarities)
        
        all_scores = torch.stack(all_scores)  # [B, N]
        
        # 使用STE生成topk mask
        mask = ste_topk_mask(all_scores, self.k)  # [B, N]
        
        # 获取topk的索引
        topk_indices = mask.nonzero(as_tuple=False).view(B, self.k, 2)[:, :, 1]  # [B, k]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.projected_dim)
        
        # 提取稀疏tokens
        sparse_image_tokens = torch.gather(activated_features, 1, topk_indices_expanded)  # [B, k, C_proj]
        
        return topk_indices, sparse_image_tokens, topk_indices_expanded

    def forward(self, image_features, hw_shapes=None, gt_info=None):
        """
        Args:
            image_features: 输入特征 [B, N, C]
            hw_shapes: 特征图尺寸列表
            gt_info: GT信息字典，包含gt_features等
        """
        B, N, C_in = image_features.shape
        identity = image_features

        # --- 1. 降维和激活 ---
        projected_features = self.down_project(image_features)
        activated_features = self.dropout(self.nonlinear_activation(projected_features))  # [B, N, C_proj]

        # --- 2. 基于GT信息更新原型 ---
        if gt_info is not None and hw_shapes is not None:
            self.update_prototypes_with_gt(activated_features, hw_shapes, gt_info)

        # --- 3. 基于原型选择tokens ---
        topk_indices, sparse_image_tokens, topk_indices_expanded = self.select_tokens_with_prototypes(activated_features)

        # --- 4. 原型与稀疏tokens的交互 ---
        queries = self.m_queries.expand(B, -1, -1)  # [B, m, C_proj]
        combined_sequence = torch.cat([queries, sparse_image_tokens], dim=1)  # [B, m+k, C_proj]
        
        # 交互: Norm -> Self-Attention -> Add
        attention_input = self.norm(combined_sequence)
        attention_output, _ = self.interaction_attention(
            query=attention_input, key=attention_input, value=attention_input
        )
        enhanced_sequence = combined_sequence + attention_output
        
        # 分离出增强后的原型和稀疏tokens
        enhanced_queries = enhanced_sequence[:, :self.m, :]  # [B, m, C_proj]
        enhanced_sparse_tokens = enhanced_sequence[:, self.m:, :]  # [B, k, C_proj]

        # --- 5. 原型更新（EMA平滑）---
        with torch.no_grad():
            # 使用增强后的原型更新query参数
            self.m_queries.copy_(
                self.prototype_update_momentum * self.m_queries + 
                (1 - self.prototype_update_momentum) * enhanced_queries.mean(dim=0, keepdim=True)
            )
        
        # --- 6. 信息还原与升维 ---
        # 创建零张量并填充增强后的tokens
        delta_x_projected = torch.zeros_like(activated_features)
        delta_x_projected.scatter_(1, topk_indices_expanded, enhanced_sparse_tokens)
        
        # 升维
        delta_x = self.up_project(delta_x_projected)  # [B, N, C_in]
        
        return identity + delta_x * self.gamma
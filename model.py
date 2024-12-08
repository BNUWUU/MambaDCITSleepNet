class MambaSleepNet_DCT(nn.Module):
    
    def __init__(self, config):
        super(MambaSleepNet_DCT, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_1 = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder)])

        # self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        # self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model)

        self.position_multi = PositionalEncoding(d_model=config.dim_model, dropout=0.1)
        # encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        # self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)
        self.transformer_encoder_multi = nn.ModuleList([Block_mamba_dct(dim=128, mlp_ratio=0.3, drop_path=0.2, cm_type='EinFFT') for _ in range(config.num_encoder_multi)])

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

        self.dct_layer = FcaBasicBlock(29, 29) # 29是时间维度

        

    def forward(self, x):
        x1 = x[:, 0, :, :]
        # x2 = x[:, 1, :, :]
        # x3 = x[:, 2, :, :]
        
        x1 = self.position_single(x1)
        # x2 = self.position_single(x2)
        # x3 = self.position_single(x3)

        for block in self.transformer_encoder_1:
            x1 = block(x1, 1, 29)     # (batch_size, 29, 128), (batch, time, frequency)
        # x2 = self.transformer_encoder_2(x2)
        # x3 = self.transformer_encoder_3(x3)

        # x = torch.cat([x1, x2, x3], dim=2)
        x = self.dct_layer(x1)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)

        for block in self.transformer_encoder_multi:
            x = block(x, 1, 29)

        x = self.layer_norm(x + residual)       # residual connection
        

        x = x.view(x.size(0), -1) # 这里增加辅助分类器的特征
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def get_middle_feature(self, x):
        x1 = x[:, 0, :, :]
        # x2 = x[:, 1, :, :]
        # x3 = x[:, 2, :, :]
        
        x1 = self.position_single(x1)
        # x2 = self.position_single(x2)
        # x3 = self.position_single(x3)
        t1 = x1
        for block in self.transformer_encoder_1:
            x1 = block(x1, 1, 29)     # (batch_size, 29, 128), (batch, time, frequency)
        # x2 = self.transformer_encoder_2(x2)
        # x3 = self.transformer_encoder_3(x3)
        t2 = x1
        # x = torch.cat([x1, x2, x3], dim=2)
        x = self.dct_layer(x1)
        t3 = x
        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)

        for block in self.transformer_encoder_multi:
            x = block(x, 1, 29)

        x = self.layer_norm(x + residual)       # residual connection


        x = x.view(x.size(0), -1) # 这里增加辅助分类器的特征
        t4 = x 
        x = self.fc1(x)
        t5 = x
        x = self.fc2(x)
        return (t1.view(x.size(0), -1), t2.view(x.size(0), -1), t3.view(x.size(0), -1),\
                 t4.view(x.size(0), -1), t5)
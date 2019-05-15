class Self_Attn(nn.Module):
    """ Self Attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x):
        """
            inputs:
                x: input feature maps(N, C, H, W)
            returns:
                out: self attention value + input feature
                attention: (C, W*H, W*H)
        """
        N, C, H, W = x.size()
        proj_query = self.query_conv(x).view(N, -1, H*W).permute(0, 2, 1) # (N, H*W, C)
        proj_key = self.key_conv(x).view(N, -1, H*W) # (N, C, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(N, -1, H*W) # (N, C, H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)

        out = self.gamma * out + x
        return out, attention
import torch
from torch import nn
class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors,latent_dim,beta):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        # print("z_flattened.shape:",z_flattened.shape)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
        # print("d:",d.shape)
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        # print("embedding.weight:",self.embedding.weight.shape)
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss

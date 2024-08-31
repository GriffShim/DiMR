import torch
import torch.nn.functional as F
from torchvision import utils
import torch.optim as optim
import matplotlib.pyplot as plt
from .model import DiMR
from .simpDiff.main import simpDiff
from .simpDiff.schedule import CosineScheduler


# Load Data
train_loader = None


# Initialize diffusion framework
cosine_scheduler = CosineScheduler()
diff = simpDiff(schedule=cosine_scheduler)


# Instantiate model
model = DiMR([48, 192, 384], [15, 8, 8])


# Training Loop
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for idx, (inputs, class_cond) in enumerate(train_loader):
    noised, noise, logSNR = diff.diffuse(inputs)

    pred_epsilon = model(noised, logSNR)

    loss = F.mse_loss(pred_epsilon, noise, reduction="none").mean(dim=[1, 2, 3])
    weighted_loss = (loss * logSNR.exp()).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if idx % 1000 == 0:
        with torch.no_grad():
            test_imgs = inputs[0].repeat(11, 1, 1, 1)
            test_noised, _, test_logSNR = diff.diffuse(test_imgs)
            test_denoised = diff.undiffuse(
                test_noised, test_logSNR, model(test_noised, test_logSNR)
            )
            grid = (
                utils.make_grid(
                    torch.cat([test_noised, test_denoised], axis=0).cpu().detach(),
                    nrow=11,
                )
                .add_(1)
                .div_(2)
                .clamp(0, 1)
            )
            grid = grid.permute(1, 2, 0)
            plt.axis("off")
            plt.imshow(grid)
            plt.show()

            sample = diff.sample(model, (3, 64, 64)).add_(1).div_(2).clamp(0, 1)
            plt.imshow(sample)
            plt.show()

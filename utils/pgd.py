import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from utils.conf import set_random_seed
from datasets.seq_cifar10 import SequentialCIFAR10, base_path
from backbone.ResNet18 import resnet18
import pandas as pd

# Configuration
use_cuda = True
set_random_seed(10)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output, _, _, _ = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct


def _pgd_whitebox(
    model,
    X,
    y,
    epsilon=0.031,
    num_steps=20,
    step_size=0.003,
    random=True,
    condition_on_correct=False,
):

    out, _, _, _ = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    if condition_on_correct:
        select_idx = out.data.max(1)[1] == y.data
        X = X[select_idx]
        y = y[select_idx]

    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out_pdg, _, _, _ = model(X_pgd)
            loss = nn.CrossEntropyLoss()(out_pdg, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    out_pdg, _, _, _ = model(X_pgd)

    err_pgd = (out_pdg.data.max(1)[1] != y.data).float().sum()
    attacks_conducted = X.shape[0]

    return err, err_pgd, attacks_conducted


def eval_adv_test_whitebox(
    model,
    device,
    test_loader,
    out_dir,
    num_steps=20,
    epsilon=0.031,
    step_size=0.003,
    random=True,
    condition_on_correct=False,
):
    """
    evaluate model by white-box attack
    """
    # dictionary for saving output
    rob_analysis_dict = dict()
    rob_analysis_dict['epsilon'] = [epsilon]
    rob_analysis_dict['num_steps'] = [num_steps]
    rob_analysis_dict['step_size'] = [step_size]

    model.eval()
    num_attacks_total = 0
    robust_err_total = 0
    natural_err_total = 0
    total = len(test_loader.dataset)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, num_attacks = _pgd_whitebox(
            model,
            X,
            y,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
            random=random,
            condition_on_correct=condition_on_correct,
        )
        robust_err_total += err_robust
        natural_err_total += err_natural
        num_attacks_total += num_attacks

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_attacks = num_attacks_total

    nat_acc = (total - nat_err) / total
    rob_acc = (total_attacks - successful_attacks) / total_attacks

    print("natural_err_total: ", nat_err)
    print("Successful Attacks: ", successful_attacks)
    print("Total Attacks:", total_attacks)

    print("Accuracy:", nat_acc)
    print("Robustness:", rob_acc)

    rob_analysis_dict['nat_err'] = [nat_err]
    rob_analysis_dict['successful_attacks'] = [successful_attacks]
    rob_analysis_dict['total_attacks'] = [total_attacks]
    rob_analysis_dict['nat_acc'] = [nat_acc]
    rob_analysis_dict['rob_acc'] = [rob_acc]
    df = pd.DataFrame(rob_analysis_dict)
    df.to_csv(os.path.join(out_dir, 'pgd_output.csv'), index='epsilon')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch PGD Attack Evaluation")
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--epsilon", default=0.031)
    parser.add_argument("--num-steps", default=20)
    parser.add_argument("--step-size", default=0.003)
    parser.add_argument("--condition_on_correct", action="store_true", default=False)
    parser.add_argument("--random", default=True)
    parser.add_argument("--model-path")
    parser.add_argument("--source-model-path")
    parser.add_argument("--target-model-path")

    args = parser.parse_args()

    # set up data loader
    test_transform = transforms.Compose(
        [transforms.ToTensor(), SequentialCIFAR10.get_normalization_transform()])

    testset = torchvision.datasets.CIFAR10('/data/output/prashant.bhat/mammoth/CIFAR10', train=False, download=False,
                                           transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, **kwargs)

    # load model
    out_dir = '/data/output/prashant.bhat/mammoth/tensorboard_runs/class-il/seq-cifar10/er/buf_500/20210817_115216_332911'
    chkpt = os.path.join(out_dir, 'checkpoint.pth')
    state_dict = torch.load(chkpt, map_location=device)
    model = resnet18(10)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("pgd white-box attack")
    eval_adv_test_whitebox(
        model,
        device,
        test_loader,
        out_dir,
        args.num_steps,
        args.epsilon,
        args.step_size,
        args.random,
        args.condition_on_correct
    )
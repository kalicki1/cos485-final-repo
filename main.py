import argparse
import os
import datetime
import pathlib

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from utils import AudioDataset, emphasis

# Hot switch which (GAN) model you would like to import 
from model_segan import Generator, Discriminator
# from model_sagan import Generator, Discriminator


# Don't overwrite existing loss_train files. 
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('-t', '--trial-name', default='default_trial', type=str, help='Where to save results')
    parser.add_argument('-d', '--data_set', default='5000', type=str, help='Name of folder in "datasets" folder with Dataset')
    parser.add_argument('-b', '--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('-e', '--num_epochs', default=86, type=int, help='train epochs number')

    # Get arguments
    opt = parser.parse_args()
    TRIAL_NAME = opt.trial_name
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    DATA_SET = opt.data_set

    # Train Loss path
    # This can change during the lifetime of the program! see program below
    TRAIN_LOSS_PATH = 'results/' + TRIAL_NAME + '/loss_train.txt' 

    # Create paths 
    pathlib.Path('epochs/{}'.format(TRIAL_NAME)).mkdir(parents=True, exist_ok=True) 
    pathlib.Path('results/{}'.format(TRIAL_NAME)).mkdir(parents=True, exist_ok=True) 

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train', dataset_name=DATA_SET)
    test_dataset = AudioDataset(data_type='test', dataset_name=DATA_SET)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    # Create file for saving training losses per epoch
    while os.path.exists(TRAIN_LOSS_PATH):
        TRAIN_LOSS_PATH = uniquify(TRAIN_LOSS_PATH)
    print(f'{TRAIN_LOSS_PATH} has been created. Writing new data...')
    with open(TRAIN_LOSS_PATH, 'w') as f:
        f.write("Epoch\td_clean_loss\td_noisy_loss\tg_loss\tg_conditional_loss\tdate\ttime\n")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0
            noisy_loss.backward()

            # d_loss = clean_loss + noisy_loss
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            g_loss.backward()
            g_optimizer.step()

            train_bar.set_description(
                'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.data.item(), noisy_loss.data.item(), g_loss.data.item(), g_cond_loss.data.item()))
            # train_bar.set_description(
            #     'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
            #         .format(epoch + 1, clean_loss.data[0], noisy_loss.data[0], g_loss.data[0], g_cond_loss.data[0]))

        # Record loss for current epoch
        with open(TRAIN_LOSS_PATH, 'a') as f:
            f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % 
                    (epoch + 1, 
                    clean_loss.data.item(), 
                    noisy_loss.data.item(), 
                    g_loss.data.item(), 
                    g_cond_loss.data.item(),
                    datetime.datetime.now()))

        # TEST model
        test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        for test_file_names, test_noisy in test_bar:
            z = nn.init.normal(torch.Tensor(test_noisy.size(0), 1024, 8))
            if torch.cuda.is_available():
                test_noisy, z = test_noisy.cuda(), z.cuda()
            test_noisy, z = Variable(test_noisy), Variable(z)
            fake_speech = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
            fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

            for idx in range(fake_speech.shape[0]):
                generated_sample = fake_speech[idx]
                file_name = os.path.join('results/' + TRIAL_NAME,
                                         '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
                wavfile.write(file_name, sample_rate, generated_sample.T)

        # Save the first and every 5th and 10th epoch's model parameters 
        if (epoch+1) == 1 or (epoch+1) % 20 == 0:
            g_path = os.path.join('epochs/' + TRIAL_NAME, 'generator-{}.pkl'.format(epoch + 1))
            d_path = os.path.join('epochs/' + TRIAL_NAME, 'discriminator-{}.pkl'.format(epoch + 1))
            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)

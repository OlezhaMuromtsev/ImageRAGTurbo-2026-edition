from model.train.losses import loss_adv_D, loss_adv_G, loss_distill, loss_elpips
from model.train.discriminator import Discriminator
from utils.text_dataset import LAIONAestheticDataset
from model.RAG import RAG
from model.text_encoder import TextEncoder
import random
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, **params):
        self.alpha_ = params['alpha']
        self.beta_ = params['beta']
        self.epochs_ = params['epochs']
        self.lr = params['lr']
        self.student_steps = params['student_steps']
        self.alphas = params['alphas'] 
        self.sigmas = params['sigmas']
        self.n_steps = params['teacher_steps']
        self.latent_size = params['latent_size']
        self.db_path = params['db_path']

    def train(self, teacher, student):
        rag = RAG(
            db_dir = self.db_path,
            device = 'cuda',
        )
        text_encoder = TextEncoder()
        text_dataset = LAIONAestheticDataset(max_prompts=1000, split="train")
        discriminator = Discriminator(encoder=teacher.encoder) 
        opt_D = optim.Adam(discriminator.projections.parameters(), lr=self.lr)
        opt_G = optim.Adam(student.adapter.parameters() + student.decoder.parameters(), lr=self.lr)
        student_steps = self.student_steps 
        for epoch in range(self.epochs_):
            for prompt in text_dataset.get_prompts_batch():
                batch_size = prompt.shape[0]
                z_T = torch.randn(batch_size, self.latent_size, self.latent_size) # сгенерировали батч шумов под текст

                # ученик очищает от шума латент z_T
                encoder_weight = 1.0
                query_embedding, text_embedding = text_encoder.forward(prompt)
                rag_out = rag.retrieve(query_embedding, top_k = 1)
                rag_cond = rag_out[0].cond
                rag_latent = rag_out[0].latent
                rag_h_out = student.forward(
                    text_embedding=rag_cond.unsqueeze(0),
                    latent=rag_latent.unsqueeze(0),
                    timestep=0,
                )
                rag_h = rag_h_out.hidden_states
                target_h = student.forward(text_embedding, timestep=self.student_timesteps, latent=z_T)
                target_h_blended = student.blending(rag_h, target_h, encoder_weight)
                z_0_fake = student.decoder(target_h_blended).latent

                # учитель очищает от шума латент
                z_0_real = teacher.forward(prompt, timestep=self.teacher_timesteps, latent=z_T)

                # teacher и student, используя подсказку из prompt, очистили от шума наш latent
                s = random.randint(0, self.n_steps)
                alpha_s = self.alphas[s]
                sigma_s = self.sigmas[s] # для всего батча параметры шума одинаковые
                eps = torch.randn_like(z_0_real)
                z_s_real = alpha_s * z_0_real + sigma_s * eps
                z_s_fake = alpha_s * z_0_fake + sigma_s * eps # в z_0_fake и z_0_real находятся якобы свободные от шума эмбеддинги из Z-space, их можно зашумить и скормить энкодеру учителя
                score_real = discriminator.forward(prompt, timestep=self.teacher_timesteps, latent=z_s_real) 
                score_fake = discriminator.forward(prompt, timestep=self.teacher_timesteps, latent=z_s_fake) # получили сконкатенированные вектора
                loss_D = loss_adv_D(score_real, score_fake)
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step() # обучили проекции дискрминатора на батче
                
                # теперь для обновленного дискриминатора поучим адаптер студента (из статьи не ясно, на том же зашумленном латенте учим или другом, сделал пока другой)
                eps = torch.randn_like(z_0_real)
                z_s_fake = alpha_s * z_0_fake + sigma_s * eps
                score_fake_updated = discriminator.forward(prompt, timestep=self.teacher_timesteps, latent=z_s_fake)
                loss_1 = loss_adv_G(score_fake_updated)
                # также учим адаптер студента и часть его декодера denoisить latent как это делает учитель
                loss_2 = loss_distill(z_0_fake, z_0_real)
                loss_3 = loss_elpips(z_0_fake, z_0_real)
                loss_G = loss_1 + self.alpha_ * loss_2 + self.beta_ * loss_3
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step() 
        return student.state_dict(), discriminator.state_dict()
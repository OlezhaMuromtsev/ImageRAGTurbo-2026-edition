

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
    
    def get_prompts(): # generator that return sequence of train prompts
        return []
    
    def train(self, teacher, student):
        encoder = teacher.get_encoder()
        discriminator = Discriminator(encoder=encoder) 
        opt_D = optim.Adam(discriminator.projections.parameters(), lr=self.lr)
        opt_G = optim.Adam(student.adapter.parameters() + student.decoder.parameters(), lr=self.lr)
        student_steps = self.student_steps 
        for epoch in range(self.epochs_):
            for prompt in self.get_prompts():
                z_0_real = teacher.forward(prompt) 
                z_0_fake = student.forward(prompt, steps=student_steps) 
                s = random.randint(1, self.n_steps) 
                alpha_s = self.alphas[s]
                sigma_s = self.sigmas[s]
                eps = torch.randn_like(z_0_real)
                z_s_real = alpha_s * z_0_real + sigma_s * eps
                z_s_fake = alpha_s * z_0_fake + sigma_s * eps
                score_real = discriminator.forward(z_s_real)
                score_fake = discriminator.forward(z_s_fake)
                loss_D = loss_adv_D(score_real, score_fake)
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                score_fake_updated = discriminator.forward(z_s_fake)
                loss_1 = loss_adv_G(score_fake_updated)
                loss_2 = loss_distill(z_0_fake, z_0_real)
                loss_3 = loss_lpips(z_0_fake, z_0_real)
                loss_G = loss_1 + self.alpha_ * loss_2 + self.beta_ * loss_3
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step() 
        return student.state_dict(), discriminator.state_dict()
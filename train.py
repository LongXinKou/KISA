import torch
from tqdm import tqdm

from utils import Calculate_Constrastive_Loss, Calculate_gtscore_Loss, cross_entropy_loss, cvae_loss

# ===========================video encoder==================================
def train_epoch(args, train_loader, model, optimizer, lr_scheduler, iteration, logger, slFeature):
    iters = len(train_loader)
    total_loss = 0
    n_batch = iteration * iters 

    with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
        if args.version == 0:
            for step, (frames,labeles,captions) in enumerate(tqdmDataLoader):
                frames = frames.to(args.device) #tensor(bs,t,3,h,w)
                # labels:tensor(bs,t,1), caption:list(bs,t)

                # get video feature and instruction feature
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)
                
                # (InforNCE objective)
                loss = Calculate_Constrastive_Loss(vFeature, cFeature, slFeature)
                # loss = Calculate_Contrastive_Loss_Extended(vFeature, cFeature, slFeature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(iteration + step/iters)  

                # logger.add_scalar('avgbatchloss', loss.item(), n_batch)
                total_loss += loss.item()
                n_batch += 1
        elif args.version == 1:
            for step, (frames,labeles,captions,gtscores) in enumerate(tqdmDataLoader):
                frames = frames.to(args.device) #tensor(bs,t,3,h,w)
                gtscores = gtscores.to(args.device) #tensor(bs,t,1)
                # labels:tensor(bs,t,1), caption:list(bs,t)

                # get video feature and instruction feature
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)
                
                # (InforNCE objective)
                loss1 = Calculate_Constrastive_Loss(vFeature, cFeature, slFeature)
                # loss1 = Calculate_Contrastive_Loss_Extended(vFeature, cFeature, slFeature)
                loss2 = Calculate_gtscore_Loss(vFeature, cFeature, gtscores)
                # loss = loss1 + args.alpha * loss2
                loss = (1-args.alpha) * loss1 + args.alpha * loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(iteration + step/iters)  

                # logger.add_scalar('avgbatchloss', loss.item(), n_batch)
                total_loss += loss.item()
                n_batch += 1
        elif args.version == 2:
            for step, (frames,labeles,captions,gtscores) in enumerate(tqdmDataLoader):
                frames = frames.to(args.device) #tensor(bs,t,3,h,w)
                gtscores = gtscores.to(args.device) #tensor(bs,t,1)
                # labels:tensor(bs,t,1), caption:list(bs,t)

                # get video feature and instruction feature
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)
                
                loss = Calculate_gtscore_Loss(vFeature, cFeature, gtscores)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(iteration + step/iters)  

                # logger.add_scalar('avgbatchloss', loss.item(), n_batch)
                total_loss += loss.item()
                n_batch += 1

    avg_loss = total_loss / iters
    logger.add_scalar('avgepochloss', avg_loss, iteration)
    print(f'{iteration}Train_Epochloss: {avg_loss}')

def test_epoch(args, test_loader, model, iteration, logger, slFeature):
    test_total_loss = 0
    if args.version == 0:
        for steps, (frames,labeles,captions) in enumerate(test_loader):
            frames = frames.to(args.device) #tensor(bs,t,3,h,w)
            # labels:tensor(bs,t,1), caption:list(bs,t), instruction:list(bs,1)

            with torch.no_grad():
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)
                
            # (InforNCE objective)
            loss = Calculate_Constrastive_Loss(vFeature, cFeature, slFeature)

            test_total_loss += loss.item()
    elif args.version == 1:
        for step, (frames,labeles,captions,gtscores) in enumerate(test_loader):
            frames = frames.to(args.device) #tensor(bs,t,3,h,w)
            gtscores = gtscores.to(args.device) #tensor(bs,t,1)

            with torch.no_grad():
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

            loss1 = Calculate_Constrastive_Loss(vFeature, cFeature, slFeature)
            loss2 = Calculate_gtscore_Loss(vFeature, cFeature, gtscores)
            # loss = loss1 + args.alpha * loss2
            loss = (1-args.alpha) * loss1 + args.alpha * loss2
            
            test_total_loss += loss.item()
    elif args.version == 2:
        for step, (frames,labeles,captions,gtscores) in enumerate(test_loader):
            frames = frames.to(args.device) #tensor(bs,t,3,h,w)
            gtscores = gtscores.to(args.device) #tensor(bs,t,1)

            with torch.no_grad():
                vFeature, cFeature = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

            loss = Calculate_gtscore_Loss(vFeature, cFeature, gtscores)

    avg_test_loss = test_total_loss / len(test_loader)
    logger.add_scalar('Test_EpochLoss', avg_test_loss, iteration)
    print(f'{iteration} Test_EpochLoss: {avg_test_loss}')
    

# ===========================skill predictor==================================
def train_pred_epoch(args, train_loader, model, predictor_model, optimizer, lr_scheduler, iteration, logger, skill_library=None, criterion=None):
    iters = len(train_loader)
    total_loss = 0
    n_batch = iteration * iters
    if args.version == 0:
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for step, (frames,labeles,captions) in enumerate(tqdmDataLoader):
                frames = frames.to(args.device) #tensor(bs,t,3,h,w)
                labels = labeles.to(args.device) #tensor(bs,t,1)

                # video encoder(freeze)
                with torch.no_grad():
                    vFeature, _ = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

                # predict skill
                if args.predictor == 'mlp':
                    pred_skill = predictor_model(vFeature=vFeature) #tensor(b,t,n)
                    loss = cross_entropy_loss(pred_skill, labels)
                elif args.predictor == 'cvae':
                    reconstructed_skill, mu, logvar = predictor_model(vFeature=vFeature, labels=labels) #tensor(b,t,1)
                    loss = cvae_loss(reconstructed_skill, labels, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(iteration + step/iters)  

                # logger.add_scalar('avgbatchloss', loss.item(), n_batch)
                total_loss += loss.item()
                n_batch += 1
    else:
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for step, (frames,labeles,captions,gtscores) in enumerate(tqdmDataLoader):
                frames = frames.to(args.device) #tensor(bs,t,3,h,w)
                labels = labeles.to(args.device) #tensor(bs,t,1)

                # video encoder(freeze)
                with torch.no_grad():
                    vFeature, _ = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

                # predict skill
                if args.predictor == 'mlp':
                    pred_skill = predictor_model(vFeature=vFeature) #tensor(b,t,n)
                    loss = cross_entropy_loss(pred_skill, labels)
                elif args.predictor == 'cvae':
                    reconstructed_skill, mu, logvar = predictor_model(vFeature=vFeature, labels=labels) #tensor(b,t,1)
                    loss = cvae_loss(reconstructed_skill, labels, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(iteration + step/iters)  

                # logger.add_scalar('avgbatchloss', loss.item(), n_batch)
                total_loss += loss.item()
                n_batch += 1
    avg_loss = total_loss / iters
    logger.add_scalar('avgepochloss', avg_loss, iteration)
    print(f'{iteration}Train_Epochloss: {avg_loss}')

def test_pred_epoch(args, test_loader, model, predictor_model, iteration, logger, skill_library=None, criterion=None):
    test_total_loss = 0
    if args.version == 0:
        for step, (frames,labeles,captions) in enumerate(test_loader):
            frames = frames.to(args.device) #tensor(bs,t,3,h,w)
            labels = labeles.to(args.device) #tensor(bs,t,1)

            with torch.no_grad():
                # video encoder(freeze)
                vFeature, _ = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

                # predict skill
                if args.predictor == 'mlp':
                    pred_skill = predictor_model(vFeature=vFeature) #tensor(b,t,n)
                    loss = cross_entropy_loss(pred_skill, labels)
                elif args.predictor == 'cvae':
                    reconstructed_skill, mu, logvar = predictor_model(vFeature=vFeature, labels=labels) #tensor(b,t,1)
                    loss = cvae_loss(reconstructed_skill, labels, mu, logvar)
            test_total_loss += loss.item()
    else:
        for step, (frames,labeles,captions,gtscores) in enumerate(test_loader):
            frames = frames.to(args.device) #tensor(bs,t,3,h,w)
            labels = labeles.to(args.device) #tensor(bs,t,1)

            with torch.no_grad():
                # video encoder(freeze)
                vFeature, _ = model(frames, caption=captions) #tensor(b,t,c)/(b,t,c)

                # predict skill
                if args.predictor == 'mlp':
                    pred_skill = predictor_model(vFeature=vFeature) #tensor(b,t,n)
                    loss = cross_entropy_loss(pred_skill, labels)
                elif args.predictor == 'cvae':
                    reconstructed_skill, mu, logvar = predictor_model(vFeature=vFeature, labels=labels) #tensor(b,t,1)
                    loss = cvae_loss(reconstructed_skill, labels, mu, logvar)
            test_total_loss += loss.item()

    avg_test_loss = test_total_loss / len(test_loader)
    logger.add_scalar('Test_EpochLoss', avg_test_loss, iteration)
    print(f'{iteration} Test_EpochLoss: {avg_test_loss}')
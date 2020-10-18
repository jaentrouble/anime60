def lr_no_update(epoch, lr):
    return 0.0

def lr_step(epoch, lr):
    if epoch<=10:
        lr = 0.2
    elif epoch<=20 :
        lr = 0.02
    elif epoch<=50:
        lr = 0.01
    else :
        lr = 0.005
    return lr

def lr_step2(epoch, lr):
    if epoch <= 10 :
        lr = (epoch +1) * 0.01
    elif epoch<=20 :
        lr = 0.02
    elif epoch<=50:
        lr = 0.01
    else :
        lr = 0.005
    return lr

def lr_step3(epoch, lr):
    if epoch <= 10 :
        lr = (epoch +1) * 0.01
    elif epoch<=20:
        lr = 0.02
    elif epoch<=50:
        lr = 0.01
    else :
        lr = lr * 0.95
    return lr

def lr_step4(epoch, lr):
    if epoch <= 10 :
        lr = (epoch +1) * 0.001
    elif epoch<=20:
        lr = 0.01
    elif epoch<=50:
        lr = 0.005
    else :
        lr = lr * 0.95
    return lr

def lr_step5(epoch, lr):
    if epoch <= 10 :
        lr = (epoch +1) * 0.001
    elif epoch<=20:
        lr = 0.01
    elif epoch<=40:
        lr = 0.005
    else :
        if epoch % 5 == 0 :
            lr = lr * 0.95
    return lr

def lr_step6(epoch, lr):
    if epoch <= 5:
        lr = (epoch+1) * 0.001
    elif epoch <= 10 :
        lr = 0.01
    elif epoch <= 20 :
        lr = 0.005
    else :
        if epoch % 5 == 0 :
            lr = lr * 0.95
    return lr

def lr_mul_inv(epoch, lr):
    return 0.01 / (epoch+1)

def low_lr(epoch, lr) :
    return 1e-5
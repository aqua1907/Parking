CLASSES = ["Empty", "Occupied"]

batch_size = 64
lr = 1e-3
epochs = 100
valid_size = 0.3
num_classes = len(CLASSES)


CFG = dict(batch_size=batch_size,
           lr=lr,
           epochs=epochs,
           valid_size=valid_size,
           num_classes=num_classes,
           CLASSES=CLASSES)

import matplotlib.pyplot as plt

file_path = 'results/metrics.txt'

train_loss = []
val_loss = []
accuracy = []
precision = []
recall = []
f1_score = []

with open(file_path, 'r') as f:
    for line in f:
        parts = line.strip().split()

        print(parts)
        accuracy.append(float(parts[1]))
        precision.append(float(parts[3]))
        recall.append(float(parts[5]))
        f1_score.append(float(parts[7]))
        val_loss.append(float(parts[9]))
        train_loss.append(float(parts[11]))


epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train/val loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, accuracy, label='Accuracy', color='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(epochs, precision, label='Precision', color='m')
plt.plot(epochs, recall, label='Recall', color='c')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision/Recall')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, f1_score, label='F1-score', color='orange')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.title('F1-score')

plt.tight_layout()
plt.show()

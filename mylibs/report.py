import matplotlib.pyplot as plt

def report_summary(eval_metrics):
	epochs = len(eval_metrics)
	plt.clf()
	plt.figure(484, figsize=(16,8))

	acc = [x["acc"] for x in eval_metrics]
	loss = [x["loss"] for x in eval_metrics]
	f1 = [x["f1"] for x in eval_metrics]

	plt.subplot(131)
	plt.plot(range(1, epochs+1), loss, marker = 'o', label = "Training Loss")
	plt.legend()
	plt.title("Loss vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Loss")

	plt.subplot(132)
	plt.plot(range(1, epochs+1), acc, marker = 'o', label = "Training Accuracy")
	plt.legend()
	plt.title("Accuracy vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	plt.plot(range(1, epochs+1), f1, marker = 'o', label = "Training F1")
	plt.legend()
	plt.title("F1 score vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("F1 score")

	plt.show()
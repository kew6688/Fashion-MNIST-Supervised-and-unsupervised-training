import matplotlib.pyplot as plt

def report_epoch_summary(eval_metrics):
	epochs = len(eval_metrics)
	plt.clf()
	plt.figure(484, figsize=(16,8))

	train_acc = [x["train"]["acc"] for x in eval_metrics]
	train_loss = [x["train"]["loss"] for x in eval_metrics]
	train_f1 = [x["train"]["f1"] for x in eval_metrics]

	val_acc = [x["val"]["acc"] for x in eval_metrics]
	val_loss = [x["val"]["loss"] for x in eval_metrics]
	val_f1 = [x["val"]["f1"] for x in eval_metrics]

	plt.subplot(131)
	plt.plot(range(1, epochs+1), train_loss, marker = 'o', label = "Training Loss")
	plt.plot(range(1, epochs+1), val_loss, marker = 'o', label = "Validation Loss")
	plt.legend()
	plt.title("Loss vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Loss")

	plt.subplot(132)
	plt.plot(range(1, epochs+1), train_acc, marker = 'o', label = "Training Accuracy")
	plt.plot(range(1, epochs+1), val_acc, marker = 'o', label = "Validation Accuracy")
	plt.legend()
	plt.title("Accuracy vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	plt.plot(range(1, epochs+1), train_f1, marker = 'o', label = "Training F1")
	plt.plot(range(1, epochs+1), val_f1, marker = 'o', label = "Validation F1")
	plt.legend()
	plt.title("F1 score vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("F1 score")

	plt.show()

def report_summary(mode_metrics, mode_description):
	plt.clf()
	plt.figure(684, figsize=(16,8))

	mode_acc = [(mode, [x["val"]["acc"] for x in metric]) for mode, metric in mode_metrics.items()]
	mode_loss = [(mode, [x["val"]["loss"] for x in metric]) for mode, metric in mode_metrics.items()]
	mode_f1 = [(mode, [x["val"]["f1"] for x in metric]) for mode, metric in mode_metrics.items()]

	plt.subplot(131)
	for mode, vals in mode_loss:
		epochs = len(vals)
		plt.plot(range(1, epochs+1), vals, marker = 'o', label = mode_description[mode])
	plt.legend()
	plt.title("Validation Loss vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Loss")

	plt.subplot(132)
	for mode, vals in mode_acc:
		epochs = len(vals)
		plt.plot(range(1, epochs+1), vals, marker = 'o', label = mode_description[mode])
	plt.legend()
	plt.title("Validation Accuracy vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	for mode, vals in mode_f1:
		epochs = len(vals)
		plt.plot(range(1, epochs+1), vals, marker = 'o', label = mode_description[mode])
	plt.legend()
	plt.title("Validation F1 vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("F1 Score")

	plt.show()

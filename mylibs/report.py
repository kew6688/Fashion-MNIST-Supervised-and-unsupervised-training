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
	plt.plot(range(1, epochs+1), train_loss, label = "Training Loss")
	plt.plot(range(1, epochs+1), val_loss, label = "Validation Loss")
	plt.legend()
	plt.title("Loss vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Loss")

	plt.subplot(132)
	plt.plot(range(1, epochs+1), train_acc, label = "Training Accuracy")
	plt.plot(range(1, epochs+1), val_acc, label = "Validation Accuracy")
	plt.legend()
	plt.title("Accuracy vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	plt.plot(range(1, epochs+1), train_f1, label = "Training F1")
	plt.plot(range(1, epochs+1), val_f1, label = "Validation F1")
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
		plt.plot(range(1, epochs+1), vals, label = mode_description[mode])
	plt.legend()
	plt.title("Validation Loss vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Loss")

	plt.subplot(132)
	for mode, vals in mode_acc:
		epochs = len(vals)
		plt.plot(range(1, epochs+1), vals, label = mode_description[mode])
	plt.legend()
	plt.title("Validation Accuracy vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	for mode, vals in mode_f1:
		epochs = len(vals)
		plt.plot(range(1, epochs+1), vals, label = mode_description[mode])
	plt.legend()
	plt.title("Validation F1 vs Epochs")
	plt.xlabel("epochs")
	plt.ylabel("F1 Score")

	plt.show()

def report_test_summary(mode_metrics, mode_description):
	plt.clf()
	plt.figure(485, figsize=(16,8))

	modes = mode_metrics.keys()
	descriptions = [mode_description[mode] for mode in modes]
	acc = [metric["acc"] for metric in mode_metrics.values()]
	loss = [metric["loss"] for metric in mode_metrics.values()]
	f1 = [metric["f1"] for metric in mode_metrics.values()]

	plt.subplot(131)
	plt.title("Test Loss for each mode")
	barlist = plt.bar(range(len(modes)), loss)
	best_index = loss.index(min(loss))
	barlist[best_index].set_color('r')
	plt.xticks(range(len(modes)), modes)
	plt.xlabel("modes")
	plt.ylabel("Loss")

	plt.subplot(132)
	plt.title("Test Accuracy for each mode")
	barlist = plt.bar(range(len(modes)), acc)
	best_index = acc.index(max(acc))
	barlist[best_index].set_color('r')
	plt.xticks(range(len(modes)), modes)
	plt.xlabel("modes")
	plt.ylabel("Accuracy")

	plt.subplot(133)
	plt.title("Test F1 for each mode")
	barlist = plt.bar(range(len(modes)), f1)
	best_index = f1.index(max(f1))
	barlist[best_index].set_color('r')
	plt.xticks(range(len(modes)), modes)
	plt.xlabel("modes")
	plt.ylabel("F1 score")

	plt.show()

	print("Mode Interpretations: ")
	for mode in modes:
		print(f"    {mode}: {mode_description[mode]}")

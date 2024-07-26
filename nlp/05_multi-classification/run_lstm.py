import lstm 

dataset, history = lstm.fit(epochs=10)
model = lstm.load_model("best_model.keras")
acc = lstm.evaluate_model(model, dataset[0][1], dataset[1][1])
lstm.plot_result(history, 'loss')
lstm.plot_result(history, 'acc')

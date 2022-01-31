if __name__ == '__main__':

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_reals = np.array([np.zeros(10)] * len(train_labels))
    for i in range(len(train_labels)):
        train_reals[i] = one_hot_label(train_labels[i])

    test_reals = np.array([np.zeros(10)] * len(test_labels))
    for i in range(len(test_labels)):
        test_reals[i] = one_hot_label(test_labels[i])

    train_features = np.array([np.zeros(len(train_data[0]) ** 2)] * len(train_data))
    for i in range(len(train_data)):
        train_features[i] = np.array(train_data[i]).flatten()
        train_features[i] = train_features[i] / np.linalg.norm(train_features[i])

    test_features = np.array([np.zeros(len(test_data[0]) ** 2)] * len(test_data))
    for i in range(len(test_data)):
        test_features[i] = np.array(test_data[i]).flatten()
        test_features[i] = test_features[i] / np.linalg.norm(test_features[i])

    epochs = 100
    learning_rate = 0.3
    batch_size = 10
    hidden_layer_size = 100

    model = Network(len(train_features[0]), hidden_layer_size, len(train_reals[0]), learning_rate)

    print(model.accuracy(test_features, test_reals))
    for e in range(epochs):
        for i in range(0, len(train_features), batch_size):
            batch = np.array(train_features[i:i + batch_size])
            reals = np.array(train_reals[i:i + batch_size])
            model.fit(batch, reals)
        print(e, model.accuracy(test_features, test_reals))

    fun_with_tests(model, test_data, test_labels, test_features, test_reals, 100)

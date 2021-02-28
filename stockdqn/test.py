

def plot_train_test_by_q(train_env, test_env, Q, algorithm_name):

    # train
    pobs = train_env.reset()
    train_acts = []
    train_rewards = []
    actions = []

    for _ in tqdm(range(len(train_env.data)-1)):

        pact = Q(pobs)
        pact = torch.argmax(pact.data)
        train_acts.append(pact)

        obs, reward, done = train_env.step(pact)
        train_rewards.append(reward)
        actions.append(pact)
        pobs = obs

    train_profits = train_env.profits

    # test
    pobs = test_env.reset()
    test_acts = []
    test_rewards = []

    for _ in tqdm(range(len(test_env.data)-1)):

        pact = Q(pobs)
        pact = torch.argmax(pact.data)
        test_acts.append(pact)

        obs, reward, done = test_env.step(pact)
        test_rewards.append(reward)

        pobs = obs
        actions.append(pact)

    test_profits = test_env.profits


plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'DQN')

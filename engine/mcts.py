import math
import random

class Node:
    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visits = 0
        self.value = 0.0

def ucb(parent, child):
    return child.value / (child.visits + 1e-6) + \
           math.sqrt(math.log(parent.visits + 1) / (child.visits + 1e-6))

class MCTS:
    def __init__(self, model, prm, sims, depth):
        self.model = model
        self.prm = prm
        self.sims = sims
        self.depth = depth

    def search(self, root_state):
        root = Node(root_state)

        for _ in range(self.sims):
            node = root
            path = [node]

            while node.children:
                node = max(node.children.values(), key=lambda c: ucb(path[-1], c))
                path.append(node)

            if len(path) < self.depth:
                logits = self.model(node.state)
                action = logits.argmax(-1)
                next_state = action.unsqueeze(0)

                child = Node(next_state)
                node.children[action.item()] = child
                path.append(child)

            reward = self.prm(self.model(path[-1].state)).mean().item()

            for n in path:
                n.visits += 1
                n.value += reward

        return max(root.children, key=lambda k: root.children[k].visits)

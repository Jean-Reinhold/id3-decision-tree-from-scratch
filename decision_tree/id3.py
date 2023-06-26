import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Node:
    attribute: str = None
    label: str = None
    children: dict = field(default_factory=dict)


@dataclass
class DecisionTree:
    tree: Node = field(default_factory=Node)

    def fit(self, df: pd.DataFrame, attributes: list[str], target: str) -> None:
        assert not df.empty
        assert target in df.columns  # Target should be contained in the df
        
        attributes = list(set(df.columns) - set([target]))
        assert len(attributes) > 0  # You can only train if there are atributes

        labels = df[target].values
        if np.all(labels == labels[0]):
            return Node(label=labels[0])

        if len(attributes) == 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            return Node(label=unique_labels[np.argmax(counts)])

        best_attribute = self.choose_best_attribute(df, attributes, target)
        root = Node(attribute=best_attribute)

        attribute_values = df[best_attribute].unique()
        for value in attribute_values:
            subset = self.get_subset(df, best_attribute, value)
            if len(subset) == 0:
                unique_labels, counts = np.unique(labels, return_counts=True)
                root.children[value] = Node(label=unique_labels[np.argmax(counts)])
            else:
                new_attributes = [attr for attr in attributes if attr != best_attribute]
                root.children[value] = self.fit(
                    subset, new_attributes, target_attribute
                )

        self.tree = root
        return root
    
    def choose_best_attribute(
        self, data: pd.DataFrame, attributes: list[str], target_attribute: str
    ) -> str:
        information_gains = []
        for attribute in attributes:
            information_gains.append(
                self.calculate_information_gain(data, attribute, target_attribute)
            )
        return attributes[np.argmax(information_gains)]

    def calculate_information_gain(
        self, data: pd.DataFrame, attribute: str, target_attribute: str
    ) -> float:
        total_entropy = self.calculate_entropy(data, target_attribute)
        attribute_values = data[attribute].unique()
        weighted_entropy = 0.0
        for value in attribute_values:
            subset = self.get_subset(data, attribute, value)
            weighted_entropy += (len(subset) / len(data)) * self.calculate_entropy(
                subset, target_attribute
            )
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def calculate_entropy(self, data: pd.DataFrame, attribute: str) -> float:
        labels = data[attribute].values
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def get_subset(
        self, data: pd.DataFrame, attribute: str, value: str | int
    ) -> pd.DataFrame:
        return data[data[attribute] == value]
    
    def predict(self, x: dict):
        assert self.tree.attribute is not None # You should train the tree first

        node = self.tree
        while node.label is None:
            attribute = node.attribute
            value = x[attribute]
            if value in node.children:
                node = node.children[value]
            else:
                return None
        return node.label
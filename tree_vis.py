from sklearn import tree
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

def explain_tree(clf, indent="\t", feature_names=None, category_names=None):
    '''Generate an explanatory string from an sklearn.tree.DecisionTreeClassifier object'''
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(f"The binary tree structure has {n_nodes} nodes:")

    for i in range(n_nodes):
        if is_leaves[i]:
            node_values = values[i][0]
            node_percents = (node_values / node_values.sum() * 100).round(0).astype(int)
            largest_id = np.argmax(node_percents)
            largest_percent = node_percents[largest_id]
            largest_category = category_names[np.argmax(node_percents)] if category_names is not None else f"category {np.argmax(node_percents)}"
            print("{indent}[Leaf]  Node {node}: {perc}% are {category}.".format(indent = node_depth[i] * indent,
                                                                        node = i,
                                                                        perc = largest_percent,
                                                                        category = largest_category))
            

        else:
            print(
                "{indent}[Split] Node {node}: "
                "go to node {left} if ({feature} < {threshold}) "
                "else to node {right}.".format(
                    indent=node_depth[i] * indent,
                    node=i,
                    left=children_left[i],
                    feature = feature_names[feature[i]] if feature_names is not None else f"feature {feature[i]}",
                    threshold = round(threshold[i],2),
                    right=children_right[i],
                )
            )

class Condition:
    def __init__(self, decision_maker):
        '''The decision_maker returns a boolean representing whether the condition is met.'''
        self.decision_maker = decision_maker

    def decide(self, input):
        return self.decision_maker(input)
    
    @property
    def decision_str(self) -> str:
        '''returns a string describing the condition, e.g. "Are you an adult?" '''
        return ""

class ThresholdCondition(Condition):
    '''Represents conditions that checks if a value is over a threshold (e.g. Age < 15?)'''
    def __init__(self, feature_name, threshold):

        def decision_maker(input:pd.DataFrame):
            return input[feature_name] < threshold
        self.decision_maker = decision_maker
        self.feature_name = feature_name
        self.threshold = threshold
    
    @property
    def decision_str(self):
        return f"{self.feature_name} < {round(self.threshold, 4)}"

class DecisionTreeNode:
    def __init__(self, condition:Condition=None, value=None, left=None, right=None, text=""):
        self.condition = condition
        self.value = value
        self.left = left
        self.right = right
        self.text = text
    
    def classify(self, input):
        if self.is_leaf:
            return self.value
        elif self.condition.decide(input):
            if self.left is not None:
                return self.left.classify(input)
        elif self.right is not None:
                return self.right.classify(input)
    
    # string properties for describing the node / the tree

    @property
    def is_leaf(self):
        return self.condition is None
    
    @property
    def node_type(self):
        return "Leaf" if self.is_leaf else "Split"
    
    @property
    def condition_description(self):
        if self.is_leaf:
            return ""
        return f"go to first branch if ({self.condition.decision_str}) else to second branch."

    @property
    def description(self):
        return self.condition_description + self.text + (str(self.value) if self.is_leaf else "")

    def explain_tree(self, n_indents=0, indent="\t", out=False):
        current_explained = indent*n_indents + f"[{self.node_type}] {self.description}"
        if self.left is not None:
            left_explained = "\n" + self.left.explain_tree(n_indents = n_indents + 1, indent=indent)
        else:
            left_explained = ""
        if self.right is not None:
            right_explained = "\n" + self.right.explain_tree(n_indents = n_indents + 1, indent=indent)
        else:
            right_explained = ""
        if out:
            print(current_explained + left_explained + right_explained)
        return current_explained + left_explained + right_explained

    # methods for visualizing the tree in a diagram

    @property
    def boxed_text(self):
        '''text that goes into a box in the diagram representing the node'''
        if self.is_leaf:
            return self.value
        return f"Is {self.condition.decision_str} ?"

    def visualize_node(self, font, margin=10):
        '''PIL image of the box (along with a margin), also computes the dimensions of the box'''
        text = self.boxed_text
        left, top, right, bottom = font.getbbox(text)
        w = right - left + 2 * margin
        h = bottom - top + 2 * margin
        image = Image.new('RGB', (w, h), color="white")
        ImageDraw.Draw(image).text((w//2,h//2), text, fill='black', anchor='mm', font=font)
        ImageDraw.Draw(image).rectangle([(margin//4, margin//2), (w-margin//4, h-margin//4)],
                                        outline="black",
                                        width= int(min(margin//4, margin ** (1/2))))
        return image, {"left":left, "top":top, "right":right, "bottom":bottom, "width":w, "height":h}

    def visualize_tree(self, margin=10, arrow_height=20, font="NotoSans-Regular.ttf", font_size=20):
        '''recursively generates an PIL image of the diagram of the tree'''
        pil_font = ImageFont.FreeTypeFont(font, size=font_size)
        pil_font_small = ImageFont.FreeTypeFont(font, size= int(font_size * (2/3)))

        # get visualizations of the left & right children, and the node itself
        if self.is_leaf:
            return self.visualize_node(pil_font, margin=margin)[0]
        if self.left is not None:
            left_image = self.left.visualize_tree(margin=margin, arrow_height=arrow_height, font=font, font_size=font_size)
        else:
            left_image = Image.new('RGB', (1,1), color="white")
        if self.right is not None:
            right_image = self.right.visualize_tree(margin=margin, arrow_height=arrow_height, font=font, font_size=font_size)
        else:
            right_image = Image.new('RGB', (1,1), color="white")
        
        current_image, current_box = self.visualize_node(margin=margin, font=pil_font)

        # create a blank image to paste the three
        combined_shape = (max(left_image.size[0] + right_image.size[0] + 3*margin, current_box["width"] + 2*margin),
                          max(left_image.size[1], right_image.size[1]) + current_box["height"] + margin + arrow_height)
        combined_image = Image.new('RGB', combined_shape, color="white")

        hor_mid = combined_shape[0] // 2
        combined_image.paste(current_image, (hor_mid-current_box["width"]//2, 0))

        hor_middle_margin_left = (combined_shape[1] - margin - current_box["height"] - left_image.size[1] - arrow_height) // 2   # horizontal margin between the top parent node and the left child
        hor_middle_margin_right = (combined_shape[1] - margin - current_box["height"] - right_image.size[1] - arrow_height) // 2    # same but for right child
        vert_edge_margin = (combined_shape[0] - left_image.size[0] - right_image.size[0]) // 3    # vertical margin between the children and the edge of the image
        combined_image.paste(left_image, (vert_edge_margin,
                                          combined_shape[1] - left_image.size[1]))
        combined_image.paste(right_image, (combined_shape[0] - vert_edge_margin - right_image.size[0],
                                           combined_shape[1] - right_image.size[1]))
        
        # arrows & lines & T/F texts
        array_image = np.array(combined_image)
        larr_start = (combined_shape[0] // 2, current_box["height"])
        larr_end = (vert_edge_margin + left_image.size[0] // 2, combined_shape[1] - left_image.size[1])
        rarr_start = (combined_shape[0] // 2, current_box["height"])
        rarr_end = (combined_shape[0] - vert_edge_margin - right_image.size[0] // 2, combined_shape[1] - right_image.size[1])
        split_height = min((larr_start[1] + larr_end[1]), (rarr_start[1] + rarr_end[1])) // 2   # the height at which the two arrows split off

        array_image = cv2.line(array_image, larr_start, (larr_start[0], split_height), (0,0,0), 2)  # line starts from then parent node
        array_image = cv2.line(array_image, (larr_end[0], split_height), (rarr_end[0], split_height), (0,0,0), 2)  # then it splits
        array_image = cv2.arrowedLine(array_image, (larr_end[0], split_height), larr_end, (0,0,0), 2, tipLength = 0.1)  # into the left...
        array_image = cv2.arrowedLine(array_image, (rarr_end[0], split_height), rarr_end, (0,0,0), 2, tipLength = 0.1)  # and right arrows

        combined_image = Image.fromarray(array_image)
        ImageDraw.Draw(combined_image).text((larr_end[0], split_height), "True ", fill='black', anchor='rt', font=pil_font_small)
        ImageDraw.Draw(combined_image).text((rarr_end[0], split_height), " False", fill='black', anchor='lt', font=pil_font_small)
                
        return combined_image

def convert_from_sklearn(clf, feature_names=None, category_names=None):

    n_nodes = clf.tree_.node_count
    left_children = clf.tree_.children_left
    right_children = clf.tree_.children_right
    dtns = [DecisionTreeNode() for i in range(n_nodes)]

    # assign left and right children
    for i, left_id, right_id in zip(range(n_nodes), left_children, right_children):
        if not left_id == -1:
            dtns[i].left = dtns[left_id]
        if not right_id == -1:
            dtns[i].right = dtns[right_id]
    
    # assign feature names and thresholds
    features = clf.tree_.feature
    thresholds = clf.tree_.threshold
    values = clf.tree_.value
    if feature_names is None:
        feature_names = [f"feature {j}" for j in range(clf.tree_.threshold.shape[0])]
    for i, feature, threshold in zip(range(n_nodes), features, thresholds):
        if not feature == -2:
            condition = ThresholdCondition(feature_name = feature_names[feature], threshold = threshold)
            dtns[i].condition = condition
    
    # assgin predicted category names
    if category_names is None:
        category_names = [f"category {k}" for k in range(clf.tree_.value.shape[2])]
    for i, value in enumerate(values):
        value = value[0]
        node_percents = (value / value.sum() * 100).round(0).astype(int)
        largest_id = np.argmax(node_percents)
        largest_percent = node_percents[largest_id]
        largest_category = category_names[np.argmax(node_percents)]
        dtns[i].value = f"{largest_percent}% are {largest_category}"
    
    return dtns[0]
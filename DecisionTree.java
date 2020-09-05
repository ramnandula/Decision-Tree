// Ram Aditya Nandula
// Fall 2020, CS 4375 - Machine Learning
// Assignment 1 - Decision Trees
// DecisionTree.java

// This program implements the ID3 decision tree learning algorithm. It only handles ternary classification tasks and ternary
// valued attributes (0, 1, 2). It handles any number of attributes. The program learns from the training file and generates a tree. 
// It then uses the tree to classify the training and test instances. Finally, it compares these values to the original values and 
// displays the accuracy.

// This class executes the algorithm.

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

public class DecisionTree {

    private class TreeNode {
        String attribute;
        int classValue;
        TreeNode[] children;

        public TreeNode(int classValue) {
            this.classValue = classValue;
            this.children = new TreeNode[3];
        }
    }

    private TreeNode root;

    public DecisionTree() {
        root = null;
    }

    public void executeAlgorithm(String[] attributes, int[][] trainExamples, int[][] testExamples) {
        // learn tree and store it
        root = learnTree(root, attributes, trainExamples, trainExamples);

        // print out tree
        traverseTree(root, 0);
        System.out.println();
        
        // classify training set
        classifyInstances(attributes, root, trainExamples, 0, "training");
        System.out.println();
        
        // classify test set
        classifyInstances(attributes, root, testExamples, 0, "test");
        System.out.println();
        System.out.println();
    }

    private TreeNode learnTree(TreeNode root, String[] attributes, int[][] examples, int[][] entireExamples) {
        // no more examples, so plurality class of all examples
        if (examples.length == 0) {
            root = new TreeNode(pluralityClassification(entireExamples, entireExamples));
            return root;
        } else if (sameClassification(examples) != -1) { // same class, so return it
            root = new TreeNode(sameClassification(examples));
            return root;
        } else if (isEmpty(attributes)) { // no attributes, first check is there is plurality for current examples
            boolean tied = classTie(examples);
            if (!tied) { // there is plurality so return it
                root = new TreeNode(pluralityClassification(examples, examples));
            } else { // tie, so go to all examples and find plurality for relevant class values in current set
                root = new TreeNode(pluralityClassification(entireExamples, examples));
            }
            return root;
        } else {
            // select best atribute and update
            String attribute = selectAttribute(attributes, examples);
            root = new TreeNode(-1);
            root.attribute = attribute;

            for (int i = 0; i < 3; i++) { // for each child/attribute value of node/attribute
                int[][] subSet = retrieveExamples(examples, attribute, i, attributes); // get relevant examples

                int index = getAttributeIndex(attribute, attributes);
                attributes[index] = null; // can't reuse attribute for path

                // repeat for children
                root.children[i] = learnTree(root.children[i], attributes, subSet, entireExamples);
                attributes[index] = attribute; // allow attribute to be reused for different path
            }
        }

        return root;
    }

    // traverse the learned tree
    private void traverseTree(TreeNode node, int j) {
        // if there is a class value, append
        if (node.classValue != -1) {
            System.out.print(node.classValue);
            return;
        }

        for (int i = 0; i < 3; i++) { // for each child 
            System.out.println();
            System.out.print(generateBars(j)); // print bars in front
            System.out.print(node.attribute + " = " + i + " : "); // print the attribute and value
            j = j + 1; // one more bar until class value return 
            traverseTree(node.children[i], j); // repeat with child
            j = j - 1; // one less bar after class value return 
        }
    }

    // classifies all training/test instances using the learned decision tree
    // compares the tree value with the original/table value
    // increments count if they are same (correctly classified) 
    private void classifyInstances(String[] attributes, TreeNode node, int[][] examples, int count, String type) {
        for (int[] example : examples) {
            int classs = findTreeClass(node, example, attributes);
            if (classs == example[examples[0].length - 1]) {
                count++;
            }
        }
        
        // percent of examples correctly classified
        double accuracy = ((double) count / examples.length) * 100;

        // output information in specified fom
        System.out.print("\nAccuracy on " + type + " set (" + examples.length + " instances): ");
        System.out.printf("%.1f", accuracy);
        System.out.print("%");
    }
    

    // finds class of one example in training/test set based on the tree
    private int findTreeClass(TreeNode node, int[] example, String[] attributes) {
        if (node.classValue != -1) {
            return node.classValue; // return class
        } else {
            // find current attribute in node and corresponding value in example
            String attribute = node.attribute;
            int attributeIndex = getAttributeIndex(attribute, attributes);
            int attributeValue = example[attributeIndex];

            attributes[attributeIndex] = null; // can't reuse
            int classV = findTreeClass(node.children[attributeValue], example, attributes); // repeat for child node with value
            attributes[attributeIndex] = attribute; // can reuse in new path
            return classV;
        }
    }

    // select the attribute with the highest information gain
    // chooses the leftmost attribute in array in case of tie
    private String selectAttribute(String[] attributes, int[][] examples) {
        double maxInformationGain = -1;
        double informationGain;
        String maxAttribute = "";

        for (String attribute : attributes) {
            if (attribute != null) {
                informationGain = calculateInformationGain(attribute, attributes, examples);
                if (informationGain > maxInformationGain) {
                    maxInformationGain = informationGain;
                    maxAttribute = attribute;
                }
            }
        }

        return maxAttribute;
    }

    // calculate information gain of an attribute based on examples
    // examples are the subset of total examples for the node
    private double calculateInformationGain(String attribute, String[] attributes, int[][] examples) {
        double informationGain = 0;
        double conditionalEntropy;

        // get the distribution for the examples in terms of the attribute
        // each row corresponds to one attribute value (0, 1, 2)
        // each column corresponds to one class (0, 1, 2)
        // last row is the totals for the classes (same for each attribute)
        // last column is totals for that attribute, used for conditional entropy
        // value at [4][4] is total num of examples, used for conditional entropy
        int attributeIndex = getAttributeIndex(attribute, attributes);
        int[][] distribution = findDistribution(examples, attributeIndex);
        int index = distribution.length - 1;

        // add the entropy of the final row which is for the total set
        informationGain += calculateEntropy(distribution[index]);

        // for each attribute value, find conditional entropy and add to information gain
        for (int i = 0; i < index; i++) {
            conditionalEntropy = -1 * ((double) distribution[i][index] / distribution[index][index]) * calculateEntropy(distribution[i]);
            informationGain += conditionalEntropy;
        }

        return informationGain;
    }

    // calculates the entropy of a given distriubtion
    // distributions are in the format: num 0s, num 1s, num 2s, total nums
    private double calculateEntropy(int[] distribution) {
        double entropy = 0;
        int total = 0;

        // get total nums
        for (int i = 0; i < distribution.length - 1; i++) {
            total += distribution[i];
        }

        // calculate each term of entropy
        for (int i = 0; i < distribution.length - 1; i++) {
            if (distribution[i] != 0) { // treat log 0 as 0 instead of undefined
                entropy += -1 * ((double) distribution[i] / total) * (Math.log((double) distribution[i] / total) / Math.log(2));
            }
        }

        return entropy;
    }

    // returns relevant subset of examples given attribute and value
    private int[][] retrieveExamples(int[][] examples, String attribute, int value, String[] attributes) {
        // find the distribution of examples based on an attribute
        int attributeIndex = getAttributeIndex(attribute, attributes);
        int[][] distribution = findDistribution(examples, attributeIndex);
        int index = distribution.length - 1;
         
        // num of rows is equal to total num of examples with that attrribute value
        int[][] subSet = new int[distribution[value][index]][examples[0].length];
        int j = 0;

        // copy examples into subset
        // only copy those examples whose attribute value equals specified value
        for (int[] example : examples) {
            if (example[attributeIndex] == value) {
                subSet[j] = example;
                j++;
            }
        }

        return subSet;
    }

    // find the distribution of examples based on an attribute
    private int[][] findDistribution(int[][] examples, int index) {
        int[][] distribution = new int[4][4];
        int classIndex = examples[0].length - 1;

        // for each example in set, find the class and attribute values 
        // increment respective count in distribution
        for (int[] example : examples) {
            int attributeValue = example[index];
            int classValue = example[classIndex];
            distribution[attributeValue][classValue]++;
        }

        int length = distribution.length - 1;
        
        // determine the values for last row (total class values)
        // determine the values for last column (total attribute values)
        // determine the value for total num of examples
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                distribution[i][length] += distribution[i][j];
                distribution[length][i] += distribution[j][i];
                distribution[length][length] += distribution[i][j];
            }
        }

        return distribution;
    }

    // determines if all class values in the example set are the same
    // returns class value if same or -1 if not the same
    private int sameClassification(int[][] examples) {
        int classIndex = examples[0].length - 1;
        int firstClassValue = examples[0][classIndex];

        for (int i = 1; i < examples.length; i++) {
            if (examples[i][classIndex] != firstClassValue) {
                return -1;
            }
        }

        return firstClassValue;
    }

    // gets the most frequent class
    // breaks ties by preferring 0 to 1 to 2
    private int pluralityClassification(int[][] examples, int[][] subSet) {
        int classIndex = examples[0].length - 1;
        Set<Integer> subSetKeys = new HashSet<>();

        // determine all classes in subset
        for (int[] sub : subSet) {
            int value = sub[classIndex];
            subSetKeys.add(value);
        }

        HashMap<Integer, Integer> frequencies = new HashMap<>();

        // get frequencies for each class for all examples
        for (int[] example : examples) {
            int classValue = example[classIndex];

            if (frequencies.containsKey(classValue)) {
                frequencies.put(classValue, frequencies.get(classValue) + 1);
            } else {
                frequencies.put(classValue, 1);
            }
        }

        int maxFrequency = 0;
        int maxFrequencyValue = 3;

        // find most frequent class in examples only incuding classes in subset
        for (Entry<Integer, Integer> val : frequencies.entrySet()) {
            if (subSetKeys.contains(val.getKey())) {
                if (val.getValue() > maxFrequency) {
                    maxFrequency = val.getValue();
                    maxFrequencyValue = val.getKey();
                } else if (val.getValue() == maxFrequency && val.getKey() < maxFrequencyValue) {
                    maxFrequencyValue = val.getKey();
                }

            }

        }

        return maxFrequencyValue;
    }

    // determines if there is two or more classses are equally frequent
    private boolean classTie(int[][] examples) {
        int classIndex = examples[0].length - 1;
        HashMap<Integer, Integer> frequencies = new HashMap<>();

        // get frequencies for each class for all examples
        for (int[] example : examples) {
            int classValue = example[classIndex];

            if (frequencies.containsKey(classValue)) {
                frequencies.put(classValue, frequencies.get(classValue) + 1);
            } else {
                frequencies.put(classValue, 1);
            }
        }
        
        ArrayList<Integer> values = new ArrayList<>();
        
        // add frequencies to list and sort
        for (Entry<Integer, Integer> val : frequencies.entrySet()) {
            values.add(val.getValue());            
        }

        Collections.sort(values, Collections.reverseOrder()); 
        
        if (values.size() == 1) // only onle class, so no tie
            return false;
        if (values.get(0) != values.get(1)) // the most frequent and next most are different so no tie
            return false;
        else // most frequent and next most are equal so tie
            return true;
    }
    
    // gets the index of the given attribute in the array
    // returns -1 if the attribute was made null, which means it is not found
    private int getAttributeIndex(String attribute, String[] attributes) {
        int attributeIndex = -1;

        for (int i = 0; i < attributes.length; i++) {
            if (attributes[i] != null) {
                if (attributes[i].equals(attribute)) {
                    attributeIndex = i;
                }
            }
        }

        return attributeIndex;
    }

    // generates bars for each line of tree output given how many times
    private String generateBars(int numBars) {
        String bars = "";

        for (int i = 0; i < numBars; i++) {
            bars += "| ";
        }

        return bars;
    }
    
    // checks if all array values are null
    private boolean isEmpty(String[] arr) {
        for (String arr1 : arr) {
            if (arr1 != null) {
                return false;
            }
        }

        return true;
    }
        
}

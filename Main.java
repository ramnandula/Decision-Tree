// Ram Aditya Nandula

// This program implements the ID3 decision tree learning algorithm. It only handles ternary classification tasks and ternary
// valued attributes (0, 1, 2). It handles any number of attributes. The program learns from the training file and generates a tree. 
// It then uses the tree to classify the training and test instances. Finally, it compares these values to the original values and 
// displays the accuracy.

// This class load the training/test files passed as command line arguments and uses it to execute the algorithm.

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws Exception {
        // train and test files as command line args
        String trainFile = args[0];
        String testFile = args[1];
        
        int[][] trainSet = loadFile(trainFile);
        int[][] testSet = loadFile(testFile);
        String[] attributes = getAttributes(trainFile);

        DecisionTree tree = new DecisionTree();

        tree.executeAlgorithm(attributes, trainSet, testSet);
    }

    // gets attributes from first row of file
    // removes last column value which is class
    // ignores blank lines
    private static String[] getAttributes(String fileName) throws FileNotFoundException {
        File file = new File(fileName);
        Scanner sc = new Scanner(file);
        String firstRow = "";
        
        // skip blank lines until first text line which is attributes
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            if (!line.trim().isEmpty()) {
                firstRow = line;
                break;
            }
        }

        String[] tableHeadings = firstRow.split("\\s+");
        String[] attributes = Arrays.copyOf(tableHeadings, tableHeadings.length - 1);

        return attributes;
    }

    // loads file into 2d array
    // skips first line which is attributes but keeps all columns
    // each row is an example
    // each column is attribute value, last column in class value
    // ignores blank lines
    private static int[][] loadFile(String fileName) throws FileNotFoundException {
        File file = new File(fileName);
        Scanner sc = new Scanner(file);
        String firstRow = "";
        
        // skip blank lines until first text line which is attributes
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            if (!line.trim().isEmpty()) {
                firstRow = line;
                break;
            }
        }

        String[] tableHeadings = firstRow.split("\\s+");

        int numRows = 0;

        // continue after attributes row and count every non space line
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            if (!line.trim().isEmpty()) {
                numRows++;
            }
        }
        
        sc = new Scanner(file);
        int numColumns = tableHeadings.length;

        int[][] inputData = new int[numRows][numColumns];
        
        // since scanner goes back to top, repeat skipping blank lines until attributes row
        while (sc.hasNextLine()) {
            String line = sc.nextLine();   
            if (!line.trim().isEmpty()) {
                break;
            }
        }
        
        // read all numbers into array
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {
                if (sc.hasNextInt()) {
                    inputData[i][j] = sc.nextInt();
                }
            }
        }

        return inputData;
    }
    
    // this method is not used by the program but is used for the learning curve
    // it generates a list of a specified size (100, 200, 300, etc. until num of training examples)
    // each value in the list is unique and between and num of training examples
    // the list contains a random unique row index of the total 
    // the new subset is generated by copying those rows from the total examples
    
//    private static int[][] generateRandomExamples(int size, int[][] examples) {
//        ArrayList<Integer> x = new ArrayList<>();
//        Random rand = new Random();
//
//        int i = 0;
//        
//        while (i != size) {
//            int randomNum = rand.nextInt(examples.length);
//            
//            if (!x.contains(randomNum)) {
//                x.add(randomNum);
//                i++;
//            }
//        }
//        
//        int[][] subSet = new int[size][examples[0].length];
//        
//        for (int j = 0; j < size; j++) {
//            subSet[j] = examples[x.get(j)];
//        }
//        
//        return subSet;
//    }  

}

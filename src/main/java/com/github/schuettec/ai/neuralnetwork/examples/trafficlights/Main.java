package com.github.schuettec.ai.neuralnetwork.examples.trafficlights;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;

public class Main {

  public static void main(String[] args) {

    DataSet trainingSet = new DataSet(12, 5);
    trainingSet.add(new DataSetRow(new double[] {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        1, 1, 0, 0, 0
    }));
    trainingSet.add(new DataSetRow(new double[] {
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        1, 1, 1, 0, 0
    }));// 2

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 3

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        1, 1, 0, 0, 0
    }));// 4

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 5

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 6

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 7

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 8

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 9

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 10

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 11

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 12

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 13

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 14

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 15

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    }, new double[] {
        0, 0, 0, 1, 0
    }));// 16

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 17

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 18

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1
    }, new double[] {
        0, 1, 0, 0, 0
    }));// 19

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1
    }, new double[] {
        0, 0, 0, 0, 1
    }));// 20

    trainingSet.add(new DataSetRow(new double[] {
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    }, new double[] {
        1, 1, 0, 0, 0
    }));// 21

    // create perceptron neural network
    NeuralNetwork myPerceptron = new Perceptron(12, 5);
    // learn the training set
    myPerceptron.learn(trainingSet);
    // save trained perceptron
    myPerceptron.save("mySamplePerceptron.nnet");
    // load saved neural network
    NeuralNetwork loadedPerceptron = NeuralNetwork.createFromFile("mySamplePerceptron.nnet");
    // test loaded neural network
    System.out.println("Testing loaded perceptron");
    testNeuralNetwork(loadedPerceptron, trainingSet);
    unknownCaseNeuralNetwork(loadedPerceptron);

  }

  public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {
    for (DataSetRow trainingElement : testSet.getRows()) {
      neuralNet.setInput(trainingElement.getInput());
      neuralNet.calculate();
      double[] networkOutput = neuralNet.getOutput();
      System.out.print("Input: " + Arrays.toString(trainingElement.getInput()));
      System.out.println(" Output: " + Arrays.toString(networkOutput));
    }
  }

  public static void unknownCaseNeuralNetwork(NeuralNetwork neuralNet) {
    DataSetRow unknownElement = new DataSetRow();// to print only
    unknownElement.setInput(new double[] {
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });// to print only
    neuralNet.setInput(new double[] {
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });
    neuralNet.calculate();
    double[] networkOutput = neuralNet.getOutput();
    System.out.print("unknown case, input: " + Arrays.toString(unknownElement.getInput()));
    System.out.println(" output: " + Arrays.toString(networkOutput));
  }
}

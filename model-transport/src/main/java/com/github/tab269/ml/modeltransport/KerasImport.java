package com.github.tab269.ml.modeltransport;

import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

/**
 * Imports a neural net in Keras format and plots its configuration.
 * @author Thomas Bertz
 */
public class KerasImport {

    public static void main(String[] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("model-transport/src/main/resources/model.json", "model-transport/src/main/resources/model.h5");
        System.out.println(model.conf().toJson());
    }
}

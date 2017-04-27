package com.github.tab269.ml.facialemotionrecognition;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Imports a trained neural net in Keras format.
 * Reads fer2013 images as files and classifies emotions.
 * Currently not working due to different number of classes in trained net (6) and dataset (7)
 * @author Thomas Bertz
 */
public class FERClassifier {

    private static Logger log = LoggerFactory.getLogger(FERClassifier.class);

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        final int height = 48;
        final int width = 48;
        final int channels = 1;
        final int rngseed = 6;
        final Random randNumGen = new Random(rngseed);
        final int batchSize = 128;
        final int outputNum = 7;
        final int numEpochs = 5;


        final String baseDir = "E:/tiefenrausch/data/fer2013/fer2013/";
        File trainDir = new File(baseDir + "Training/");
        File testDir = new File(baseDir + "PrivateTest/");

        FileSplit fileSplitTrain = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit fileSplitTest = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMaker);
        rrTrain.initialize(fileSplitTrain);
        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMaker);
        rrTest.initialize(fileSplitTest);

        DataSetIterator iteratorTrain = new RecordReaderDataSetIterator(rrTrain, batchSize,1, outputNum);
        DataSetIterator iteratorTest = new RecordReaderDataSetIterator(rrTest, batchSize,1, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scalerTrain = new ImagePreProcessingScaler(0,1);
        scalerTrain.fit(iteratorTrain);
        iteratorTrain.setPreProcessor(scalerTrain);

        DataNormalization scalerTest = new ImagePreProcessingScaler(0,1);
        scalerTest.fit(iteratorTest);
        iteratorTest.setPreProcessor(scalerTest);


        log.info("**** Import Model ****");
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("model-transport/src/main/resources/model.json", "model-transport/src/main/resources/model.h5");
        model.init();
        model.setListeners(new ScoreIterationListener(10));


        log.info("******EVALUATE TRAINED MODEL******");
        Evaluation evaluation = new Evaluation(outputNum);
        while (iteratorTest.hasNext()) {
            DataSet ds = iteratorTest.next();
            INDArray prediction = model.output(ds.getFeatures(), false);
            evaluation.eval(ds.getLabels(), prediction);
        }
        log.info(evaluation.stats());
    }
}

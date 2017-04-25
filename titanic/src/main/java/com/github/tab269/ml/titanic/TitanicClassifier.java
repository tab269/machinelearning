package com.github.tab269.ml.titanic;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

/**
 * Classifier that trains on the titanic dataset.
 * Saves relatively good models to file.
 * @author Thomas Bertz
 */
public class TitanicClassifier {
    private static Logger log = LoggerFactory.getLogger(TitanicClassifier.class);
    private static final String modelFilenamePrefix = "titanic/trained-models/" + "TitanicClassifier";
    private static final String modelFilenameInfix = "." + new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
    private static final String modelFilenamePostfix = ".model.dl4j";
    private static final String modelFilename = modelFilenamePrefix + modelFilenameInfix + modelFilenamePostfix;

    public static void main(String[] args) throws IOException, InterruptedException {
        int skipNumLines = 1;
        int labelIndex = 1;

        int seed = 123;
        int inputHiddenFactor = 8;
        int numOutputs = 2;
        double learningRate = 0.01;
        int nEpochs = 8192;
        int nExamples = 1309;
        int batchSize = nExamples / 1;
        int iterations = (int)Math.ceil(((double)nExamples / batchSize));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnInteger("pclass")
                .addColumnInteger("survived")
                .addColumnString("name")
                .addColumnCategorical("sex", "male", "female")
                .addColumnDouble("age")
                .addColumnDouble("ageFixed")
                .addColumnInteger("sibsp")
                .addColumnInteger("parch")
                .addColumnString("ticket")
                .addColumnDouble("fare")
                .addColumnDouble("fareFixed")
                .addColumnString("cabin")
                .addColumnCategorical("embarked", "S", "C", "Q")
                .addColumnCategorical("embarkedFixed", "S", "C", "Q")
                .addColumnString("boat")
                .addColumnString("body")
                .addColumnString("home.dest")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("name", "age", "ticket", "fare", "cabin", "embarked", "boat", "body", "home.dest")
//                .categoricalToInteger("sex")
//                .categoricalToInteger("embarkedFixed")
                .categoricalToOneHot("sex")
                .categoricalToOneHot("embarkedFixed")
//                .conditionalReplaceValueTransform("age", new DoubleWritable(45.0), new DoubleColumnCondition("age", ConditionOp.Equal, Double.NaN))
                .build();
        log.info("" + tp.getFinalSchema());


        RecordReader rr = new CSVRecordReader(skipNumLines, CSVRecordReader.QUOTE_HANDLING_DELIMITER, "\"");
        String nameInputResource = "biostat.mc.vanderbilt.edu.FIXED.csv";
        rr.initialize(new FileSplit(new ClassPathResource(nameInputResource).getFile()));


        TransformProcessRecordReader tprr = new TransformProcessRecordReader(rr, tp);
        DataSetIterator allIter = new RecordReaderDataSetIterator(tprr, batchSize, labelIndex, numOutputs);
        DataSet allData = allIter.next();

//        allData.shuffle();
//        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(0.70);
        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain((int)(0.70 * batchSize), new Random(seed));
        DataSet trainDS = splitTestAndTrain.getTrain();
        DataSet testDS = splitTestAndTrain.getTest();

        int numInputs = tp.getFinalSchema().numColumns() - 1;
        int numHiddenNodes = numInputs * inputHiddenFactor;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener((int) Math.ceil((0.1 * nEpochs))));

        log.info("Train model ...");
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainDS);
        }

        log.info("Evaluate model ...");
        Evaluation evaluation = new Evaluation(numOutputs);
        INDArray prediction = model.output(testDS.getFeatures());
        evaluation.eval(testDS.getLabels(), prediction);
        System.out.println(evaluation.stats());

        if (evaluation.accuracy() >= 0.75) {
            log.info("Trained a relatively good model, saving it ...");
            File f = new File(modelFilename);
            if (!f.exists()) f.createNewFile();
            ModelSerializer.writeModel(model, f, false);
        }
    }
}

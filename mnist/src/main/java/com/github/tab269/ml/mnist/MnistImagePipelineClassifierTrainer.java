package com.github.tab269.ml.mnist;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

/**
 *  Adjusted and tuned by Thomas Bertz
 *
 *  This code example is featured in this youtube video
 *
 *  https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 *  This differs slightly from the Video Example,
 *  The Video example had the data already downloaded
 *  This example includes code that downloads the data
 *
 *  Data Source
 *  wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 *  followed by tar xzvf mnist_png.tar.gz
 *
 *  OR
 *  git clone https://github.com/myleott/mnist_png.git
 *  cd mnist_png
 *  tar xvf mnist_png.tar.gz
 *
 *  15MB  download
 *  158MB uncompressed
 *
 *  This examples builds on the MnistImagePipelineExample
 *  by Saving the Trained Network
 *
 */
public class MnistImagePipelineClassifierTrainer {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineClassifierTrainer.class);

    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
    public static final String DATA_PATH = "mnist/downloadedResources/";


    public static void main(String[] args) throws Exception {
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 5;


        downloadData();

        File trainDir = new File(DATA_PATH + "/mnist_png/training");
        File testDir = new File(DATA_PATH + "/mnist_png/testing");

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


        log.info("**** Build Model ****");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(height * width)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(100)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(iteratorTrain);
        }

        log.info("******SAVE TRAINED MODEL******");
        File d = new File("mnist/trained-models/");
        if (!d.exists()) d.mkdir();
        File locationToSave = new File("mnist/trained-models/trained_mnist_model.zip");
        log.info("to path " + locationToSave.getAbsolutePath());
        ModelSerializer.writeModel(model, locationToSave, false);


        log.info("******EVALUATE TRAINED MODEL******");
        Evaluation evaluation = new Evaluation(outputNum);
        while (iteratorTest.hasNext()) {
            DataSet ds = iteratorTest.next();
            INDArray prediction = model.output(ds.getFeatures(), false);
            evaluation.eval(ds.getLabels(), prediction);
        }
        log.info(evaluation.stats());
    }


    /*
     * ----------- Helper methods -----------
     */
    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "mnist_png";
        File extractedFile = new File(extractedPath);

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (15MB)...");
            getMnistPNG();
            //Extract tar.gz file to output directory
            extractTarGz(archizePath, DATA_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                extractTarGz(archizePath, DATA_PATH);
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }


    private static final int BUFFER_SIZE = 4096;
    private static void extractTarGz(String filePath, String outputPath) throws IOException {
        int fileCount = 0;
        int dirCount = 0;
        System.out.print("Extracting files");
        try(TarArchiveInputStream tais = new TarArchiveInputStream(
                new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(filePath))))) {
            TarArchiveEntry entry;

            /** Read the tar entries using the getNextEntry method **/
            while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
                //System.out.println("Extracting file: " + entry.getName());

                //Create directories as required
                if (entry.isDirectory()) {
                    new File(outputPath + entry.getName()).mkdirs();
                    dirCount++;
                } else {
                    int count;
                    byte data[] = new byte[BUFFER_SIZE];

                    FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
                    BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
                    while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
                        dest.write(data, 0, count);
                    }
                    dest.close();
                    fileCount++;
                }
                if (fileCount % 1000 == 0) System.out.print(".");
            }
        }

        System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
    }



    public static void getMnistPNG() throws IOException {
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";

        File f = new File(archizePath);
        if (!f.exists()) {
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            try (CloseableHttpResponse response = client.execute(new HttpGet(DATA_URL))) {
                HttpEntity entity = response.getEntity();
                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(f)) {
                        entity.writeTo(outstream);
                        outstream.flush();
                        outstream.close();
                    }
                }
            }
            System.out.println("Data downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing directory at " + f.getAbsolutePath());
        }
    }
}

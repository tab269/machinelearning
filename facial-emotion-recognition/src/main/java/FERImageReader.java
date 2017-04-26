import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.util.Arrays.stream;

/**
 * Reads images from fer2013 csv file.
 * @author Thomas Bertz
 */
public class FERImageReader {
    private static final Logger log = LoggerFactory.getLogger(FERImageReader.class);

    public static void main(String[] args) throws IOException, InterruptedException {

//        Schema inputSchema = new Schema.Builder()
//                .addColumnInteger("emotion")
//                .addColumnString("pixels")
//                .build();
//        TransformProcess tp = new TransformProcess.Builder(inputSchema)
//                .build();

        List<Double> labels = new ArrayList<>();
        List<Double> features = new ArrayList<>();
        RecordReader rrTest = new CSVRecordReader(1, ",");
        rrTest.initialize(new FileSplit(new File("E:/tiefenrausch/data/fer2013/fer2013/fer2013.PrivateTest.head-3.csv")));
        INDArray myDataExamples = Nd4j.zeros(48, 48);
        int i = 0;
        while (rrTest.hasNext()) {
            List<Writable> example = rrTest.next();
            Double label = example.get(0).toDouble();
            String pixelsAsString = example.get(1).toString();
            String[] pixelsStringArray = pixelsAsString.split(" ");
            double[] pixels = Arrays.stream(pixelsStringArray).mapToDouble(Integer::parseInt).toArray();
            INDArray featurePixels = Nd4j.create(pixels);
            featurePixels = featurePixels.reshape(48, 48);
            if (i == 0) {
                Nd4j.copy(featurePixels, myDataExamples);
            } else {
                Nd4j.hstack(myDataExamples, featurePixels);
            }
            i++;
        }
        log.info("" + myDataExamples);

//        int numClasses = 7;
//        int batchSize = 100;
//        DataSetIterator dsIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, numClasses);
//        DataSet ds = dsIter.next();
//        log.info("Label\n" + ds.getLabels());
//        log.info("Label\n" + ds.getFeatures());
    }
}

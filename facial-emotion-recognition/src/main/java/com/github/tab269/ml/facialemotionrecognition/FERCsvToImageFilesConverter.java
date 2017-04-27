package com.github.tab269.ml.facialemotionrecognition;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static java.util.Arrays.stream;

/**
 * Reads fer2013 csv file and writes single files to disk.
 * @author Thomas Bertz
 */
public class FERCsvToImageFilesConverter {
    private static final Logger log = LoggerFactory.getLogger(FERCsvToImageFilesConverter.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        int w = 48;
        int h = 48;
        int i = 0;
        String baseIODir = "E:/tiefenrausch/data/fer2013/fer2013/";
        RecordReader rrTest = new CSVRecordReader(1, ",");
        rrTest.initialize(new FileSplit(new File(baseIODir + "fer2013.csv")));
        while (rrTest.hasNext()) {
            if (i % 100 == 0) System.out.println(i);
            List<Writable> example = rrTest.next();
            String label = example.get(0).toString();
            String pixelsAsString = example.get(1).toString();
            String type = example.get(2).toString();
            String[] pixelsStringArray = pixelsAsString.split(" ");
            int[] pixels = Arrays.stream(pixelsStringArray).mapToInt(Integer::parseInt).toArray();


            BufferedImage outputImage = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            WritableRaster raster = outputImage.getRaster();
            raster.setSamples(0, 0, w, h, 0, pixels);
            File b = new File(baseIODir + type);
            if (!b.exists()) b.mkdir();
            String pathname = baseIODir + type + "/" + label + "/";
            File d = new File(pathname);
            if (!d.exists()) d.mkdir();
            ImageIO.write(outputImage, "png", new File(pathname + i + ".png"));
            i++;
        }
        System.out.println(i);
    }
}

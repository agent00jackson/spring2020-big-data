import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vectors
import spark.implicits._

def loadData() : DataFrame = {
    val rawDf = {spark.read.format("csv")
	.option("header","true")
    .option("inferSchema", "true")
	.load("/files/creditcard.csv")
    }

    val vCols = rawDf.drop("Class").drop("Time").columns

    val assembler = {new VectorAssembler()
        .setInputCols(vCols)
        .setOutputCol("vfeatures")
    }

    val scaler ={ new StandardScaler()
        .setInputCol("vfeatures")
        .setOutputCol("features")
        .setWithStd(true)
        .setWithMean(false)
    }

    val assembled = assembler.transform(rawDf).drop(vCols:_*)
    val scalerModel = scaler.fit(assembled)

    val scaled = scalerModel.transform(assembled).drop("vfeatures")

    return scaled
    //val secondAssembler = {new VectorAssembler()
    //    .setInputCols(Array("scaled","Time"))
    //    .setOutputCol("features")
    //}
//
    //val cleaned = {secondAssembler.transform(scaled)
    //    .drop("Time", "scaled")
    //}
//
    //return cleaned
}

def trainModel(trainData: DataFrame, iterations: Int, tol: Double) : LogisticRegressionModel = {
    val lr = {new LogisticRegression()
        .setMaxIter(iterations)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
        .setTol(tol)
        .setLabelCol("Class")
        .setFeaturesCol("features")
    }
    return lr.fit(trainData)
}

def RunMetrics(theModel: LogisticRegressionModel, validData: DataFrame){
    val trainingSummary = theModel.binarySummary
    println("Training:")
    //Good val above .7, perfect = 1, worst = .5
    println(s"\tAUROC: ${trainingSummary.areaUnderROC}")
    println(s"\tIterations: ${trainingSummary.totalIterations}")

    //1 is best
    println("Validation:")
    val validSummary = theModel.evaluate(validData)
    validSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tF1-Score($label) = $f")
    }
}

def resample(theData: DataFrame, ratio: Double) : DataFrame = {
    val pos = theData.filter(r => r(1) == 1)
    val neg = theData.filter(r => r(1) == 0)
    val p_count = pos.count()
    val n_count = neg.count()
    val frac = (p_count * ratio) / n_count
    val sampled = neg.sample(false, frac, 11L)
    return sampled.union(pos)
}

val data = loadData()
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val validation = splits(1)

val trainResample = resample(training, .97)

val model = trainModel(trainResample, 100, 1e-6)
RunMetrics(model, validation)
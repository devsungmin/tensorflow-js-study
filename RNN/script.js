import { MnistData } from "./data.js";

//비동기식 -> Mnist 데이터를 보여줌
async function showExamples(data) {
    // Create a container in the visor
    const surface = tfvis
        .visor()
        .surface({ name: "Input Data Examples", tab: "Input Data" });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = "margin: 4px;";
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}


//RNN Model Layer
function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;
    
    // simpleRNN
    model.add(tf.layers.simpleRNN({
        units: 128, //hidden layer cnt
        kernelInitializer: 'glorotUniform',
        returnSequences: true
    }));
   
    //model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(tf.layers.dense({
        units: chars
    }))
    model.add(tf.layers.activation({ activation: 'relu' }));
    // model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam',
        metrics: ['accuracy']
    });
}

//비동기식 -> vison을 이용해서 그래프 찍기
async function train(model, data) {
    const metrics = ["loss", "val_loss", "acc", "val_acc"];
    const container = {
      name: "Model Training",
      styles: { height: "1000px" }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // 학습 된 결과 값을 보여줌
  
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 100,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }
  
  const classNames = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine"
  ];
  
  function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1
    ]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);
  
    testxs.dispose();
    return [preds, labels];
  }
  
  // 비동기식
  async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: "Accuracy", tab: "Evaluation" };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  
    labels.dispose();
  }
  
  //비동기식
  async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: "Confusion Matrix", tab: "Evaluation" };
    tfvis.render.confusionMatrix(
      container,
      { values: confusionMatrix },
      classNames
    );
  
    labels.dispose();
  }
  
  // 학습
  async function run() {
    console.log("run...");
    const data = new MnistData();
  
    await data.load();
    await showExamples(data); // 표를 보여줌
  
    const model = getModel();
    tfvis.show.modelSummary({ name: "Model Architecture" }, model);
  
    await train(model, data);
  
    await showAccuracy(model, data);
    await showConfusion(model, data);
  }
  
  document.addEventListener("DOMContentLoaded", run);
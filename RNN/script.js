import { Data } from "./data.js"
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

async function RNN(data) {
    // 시각화 도구
    const surface = tfvis.visor().surface({ name: "Show Input Data", tab: "Input Data" });

    //RNN Model Layer
    function getModel() {
        //     model = Sequential()
        const model = tf.sequential();
        // simpleRNN
        model.add(tf.layers.simpleRNN({
            units: hiddenSize,
            returnSequences: true
        }));
        model.add(tf.layers.simpleRNN({

            returnSequences: false
        }))

        //model.add(layers.TimeDistributed(layers.Dense(len(chars))))
        model.add(tf.layers.timeDistributed(
            { layer: tf.layers.dense({ units: vocabularySize }) }));
        //adam = optimizers.Adam(lr = 0.001)
        model.add(tf.layers.activation({ activation: 'softmax' }));
        // model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: 'adam',
            metrics: ['accuracy']
        });

    }

}

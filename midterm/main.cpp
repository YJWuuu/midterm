/*

#include "mbed.h"
#include <cmath>
#include "DA7212.h"

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
EventQueue queue(32 * EVENTS_EVENT_SIZE);
Thread t;
int idC[32];
int indexC = 0;
DigitalOut green_led(LED2);



int song[42] = {
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261,
    392, 392, 349, 349, 330, 330, 294,
    392, 392, 349, 349, 330, 330, 294,
    261, 261, 392, 392, 440, 440, 392,
    349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2};


void playNote(int freq)
{
    for (int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    // the loop below will play the note for the duration of 1s
    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
        audio.spk.play(waveform, kAudioTxBufferSize);
    }
}
void playNoteC(void)
{
    idC[indexC++] = queue.call_every(1, playNote, 261);
    indexC %= 32;
}
void stopPlayNoteC(void)
{
    for (auto &i : idC) {
        queue.cancel(i);
    }
}


void playNoteC(void)
{
    for (int i = 0; i < 42; i++) {
        int length = noteLength[i];
        while (length--) {
            queue.call(playNote, song[i]);
            if (length <= 1) wait(1.0);
        }
    }
}

int main(void)
{
    t.start(callback(&queue, &EventQueue::dispatch_forever));
    sw2.fall(queue.event(playNoteC));
    sw3.fall(queue.event(stopPlayNoteC));
}

*/


#include "mbed.h"
#include "uLCD_4DGL.h"
#include <cmath>
#include "DA7212.h"

#define bufferLength (32)
#define signalLength (42)

DA7212 audio;
int sig[signalLength];
int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;
int idC[32];
int indexC = 0;
EventQueue playQ(32 * EVENTS_EVENT_SIZE);
DigitalOut green_led(LED2);

int taiko[42] = {
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2,
    1, 1, 1, 1, 1, 1, 2};

int noteLength[42];
Thread DNN;
Thread MODE;
Thread LOAD;
Thread PLAY;
Thread TAIKO;
EventQueue modeQ(32 * EVENTS_EVENT_SIZE);
EventQueue loadQ(32 * EVENTS_EVENT_SIZE);
EventQueue taikoQ(32 * EVENTS_EVENT_SIZE);
uLCD_4DGL uLCD(D1, D0, D2);
int display = 0;
int item = 0;
int song = 0;
int info = 0;
int stop = 0;
int load = 0;
int score = 0;

Serial pc(USBTX, USBRX);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ***************    INITIALIZATION OF DNN    ***************
    // Create an area of memory to use for input, output, and intermediate arrays.
    // The size of this will depend on the model you're using, and may need to be
    // determined by experimentation.
    constexpr int kTensorArenaSize = 60 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data
    bool should_clear_buffer = false;
    bool got_data = false;

    // The gesture index of the prediction
    int gesture_index;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

    static tflite::MicroOpResolver<6> micro_op_resolver;

    // Build an interpreter to run the model with
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    TfLiteTensor* model_input = interpreter->input(0);
    int input_length = model_input->bytes / sizeof(float);
    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
// *************  END OF INITIALIZATION OF DNN  *************


void uLCD_display(void)
{
    switch(info) {
        case 0:
            break;
        case 1:
            uLCD.cls();
            uLCD.printf("\n    Twinkle2,");
            uLCD.printf("\n      Little Star\n");
            break;
        case 2:
            uLCD.cls();
            uLCD.printf("\n    DoReMiFa\n\n");
            break;
        case 3:
            uLCD.cls();
            uLCD.printf("\n    SiLaSoFa\n\n");
            break;
        case 4:
            uLCD.cls();
            uLCD.printf("\n    Taiko game!\n\n");
        default:
            break;
    }
    switch(display) {
        case 0:
            uLCD.cls();
            uLCD.printf("\n     Welcome!\n\n");
            break;
        case 1:
            uLCD.cls();
            uLCD.printf("\n  Selection mode\n\n");
            uLCD.printf("\n    Forward\n\n");
            uLCD.printf("\n    Backward\n\n");
            uLCD.printf("\n    Change Songs\n\n");
            uLCD.printf("\n    Taiko game!\n\n");
            break;
        case 2:
            uLCD.cls();
            uLCD.printf("\n  Select songs\n\n");
            uLCD.printf("\n    Twinkle2,");
            uLCD.printf("\n      Little Star\n");
            uLCD.printf("\n    DoReMiFa\n\n");
            uLCD.printf("\n    SiLaSoFa\n\n");
            uLCD.printf("\n    Taiko game!\n\n");
            break;
        case 3:
            uLCD.printf("\n    Loading...\n\n");            
            break;
        default:
            break;
    }
    switch(item) {
        case 0:
            break;
        case 1:
            uLCD.locate(2, 4);
            uLCD.printf(">");
            break;
        case 2:
            uLCD.locate(2, 7);
            uLCD.printf(">");
            break;
        case 3:
            uLCD.locate(2, 10);
            uLCD.printf(">");
            break;
        case 4:
            uLCD.locate(2, 13);
            uLCD.printf(">");
            break;
        default:
            break;
    }
}

void playNote(int freq)
{
    for(int i = 0; i < kAudioTxBufferSize; i++) {
        waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
    }
    audio.spk.play(waveform, kAudioTxBufferSize);
}

void loadSignal(void)
{
    if (display == 2) {
        song = item - 1;
    }
    green_led = 0;
    pc.printf("%d\r\n", song);
    int i = 0;
    serialCount = 0;
    audio.spk.pause();
    display = 3;
    item = 0;
    info = song + 1;
    uLCD_display();
    uLCD.locate(16, 10);
    uLCD.printf("|");
    uLCD.locate(2, 10);
    while (i < signalLength) {
        if (pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if (serialCount == 3) {
                serialInBuffer[serialCount] = '\0';
                sig[i] = (int) atoi(serialInBuffer);
                if (i % 6 == 0) {
                    if (sig[0] == 0) {
                        uLCD.printf("Failed :(\n  please reset.");
                        break;
                    }
                    uLCD.printf(".");
                }
                //uLCD.printf("%d: %d\r\n", i, sig[i]);
                serialCount = 0;
                i++;
            }
        }
    }

    i = 0;
    while (i < signalLength) {
        if (pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if (serialCount == 1) {
                serialInBuffer[serialCount] = '\0';
                noteLength[i] = (int) atoi(serialInBuffer);
                if (i % 6 == 0) {
                    uLCD.printf(".");
                }
                //uLCD.printf("%d: %d   \r\n", i, noteLength[i]);
                serialCount = 0;
                i++;
            }
        }
    }
    display = 4;
    info = song + 1;
    uLCD_display();
    green_led = 1;
    for (int i = 0; i < 42 && !stop; i++) {
        int length = noteLength[i];
        while (length--) {
            for (int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
                playQ.call(playNote, sig[i]);
            }
            if (length < 1) wait(1.0);
        }
    }
    green_led = 0;
}
void uLCD_Taiko(void)
{
    score = 0;
    item = -1;
    int i = 0;
    uLCD.cls();
    uLCD.printf("\n    Taiko game!\n\n");
    uLCD.locate(2, 10);
    uLCD.printf("    score:%d\n\n", score);
    uLCD.circle(30, 50, 10, BLUE);
    uLCD.circle(60, 50, 8, GREEN);
    uLCD.circle(90, 50, 8, GREEN);
    uLCD.circle(120, 50, 8, GREEN);
    wait(0.2);
    uLCD.cls();
    uLCD.printf("\n    Taiko game!\n\n");
    uLCD.locate(2, 10);
    uLCD.printf("    score:%d\n\n", score);          
    uLCD.circle(30, 50, 10, BLUE);
    uLCD.circle(45, 50, 8, GREEN);
    uLCD.circle(75, 50, 8, GREEN);
    uLCD.circle(105, 50, 8, GREEN);
    uLCD.circle(135, 50, 8, GREEN);
    wait(0.2);
    i += 2;
    while (i / 14 < 6) {
        i += 7;
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(30, 50, 8, GREEN);
        uLCD.circle(60, 50, 8, GREEN);
        uLCD.circle(90, 50, 8, GREEN);
        uLCD.circle(120, 50, 8, RED);
        wait(0.2);
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(45, 50, 8, GREEN);
        uLCD.circle(75, 50, 8, GREEN);
        uLCD.circle(105, 50, 8, RED);
        uLCD.circle(135, 50, 8, GREEN);
        wait(0.2);
        i += 2;
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(30, 50, 8, GREEN);
        uLCD.circle(60, 50, 8, GREEN);
        uLCD.circle(90, 50, 8, RED);
        uLCD.circle(120, 50, 8, GREEN);
        wait(0.2);
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(45, 50, 8, GREEN);
        uLCD.circle(75, 50, 8, RED);
        uLCD.circle(105, 50, 8, GREEN);
        uLCD.circle(135, 50, 8, GREEN);
        wait(0.2);
        i += 2;
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(30, 50, 8, GREEN);
        uLCD.circle(60, 50, 8, RED);
        uLCD.circle(90, 50, 8, GREEN);
        uLCD.circle(120, 50, 8, GREEN);
        wait(0.2);
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(45, 50, 8, RED);
        uLCD.circle(75, 50, 8, GREEN);
        uLCD.circle(105, 50, 8, GREEN);
        uLCD.circle(135, 50, 8, GREEN);
        wait(0.2);
        i += 2;
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(30, 50, 8, RED);
        uLCD.circle(60, 50, 8, GREEN);
        uLCD.circle(90, 50, 8, GREEN);
        uLCD.circle(120, 50, 8, GREEN);
        wait(0.2);
        uLCD.cls();
        uLCD.printf("\n    Taiko game!\n\n");
        uLCD.locate(2, 10);
        uLCD.printf("    score:%d\n\n", score);
        uLCD.circle(30, 50, 10, BLUE);
        uLCD.circle(45, 50, 8, GREEN);
        uLCD.circle(75, 50, 8, GREEN);
        uLCD.circle(105, 50, 8, GREEN);
        uLCD.circle(135, 50, 8, GREEN);
        wait(0.2);
        i += 2;
    }
    uLCD.cls();
    uLCD.text_width(2);
    uLCD.text_height(2);
    uLCD.printf("\n score:\n\n   %d\n\n", score);
}
void loadTaiko(void)
{
    green_led = 0;
    pc.printf("%d\r\n", 0);
    int i = 0;
    serialCount = 0;
    audio.spk.pause();
    display = 3;
    item = 0;
    info = 4;
    uLCD_display();
    uLCD.locate(16, 10);
    uLCD.printf("|");
    uLCD.locate(2, 10);
    while (i < signalLength) {
        if (pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if (serialCount == 3) {
                serialInBuffer[serialCount] = '\0';
                sig[i] = (int) atoi(serialInBuffer);
                if (i % 6 == 0) {
                    if (sig[0] == 0) {
                        uLCD.printf("Failed :(\n  please reset.");
                        break;
                    }
                    uLCD.printf(".");
                }
                //uLCD.printf("%d: %d\r\n", i, sig[i]);
                serialCount = 0;
                i++;
            }
        }
    }

    i = 0;
    while (i < signalLength) {
        if (pc.readable()) {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;
            if (serialCount == 1) {
                serialInBuffer[serialCount] = '\0';
                noteLength[i] = (int) atoi(serialInBuffer);
                if (i % 6 == 0) {
                    uLCD.printf(".");
                }
                //uLCD.printf("%d: %d   \r\n", i, noteLength[i]);
                serialCount = 0;
                i++;
            }
        }
    }
    display = 4;
    info = 4;
    uLCD_display();
    green_led = 1;
    taikoQ.call(uLCD_Taiko);
    wait(0.9);
    for (int i = 0; i < 42 && !stop; i++) {
        int length = noteLength[i];
        while (length--) {
            for (int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j) {
                playQ.call(playNote, sig[i]);
            }
            if (length < 1) wait(1.0);
        }
    }
    green_led = 0;
}

void loadSignalHandler(void)
{
    stop = 0;
    //loadQ.call(loadSignal);

    if (item == 4) {
        loadQ.call(loadTaiko);
    }
    else {
        loadQ.call(loadSignal);
    }
    
}

// Return the result of the last prediction
int PredictGesture(float* output) {
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;
    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;

    for (int i = 0; i < label_num; i++) {
        if (output[i] > 0.8) this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1) {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict) {
        continuous_count += 1;
    } else {
        continuous_count = 0;
    }
    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,
    // report a negative result
    if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
        return label_num;
    }
    // Otherwise, we've seen a positive result, so clear all our variables
    // and report it
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}
void dnn(void)
{
    while (true) {

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
            should_clear_buffer = false;
            continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on index: %d\n", begin_index);
            continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
            //error_reporter->Report(config.output_message[gesture_index]);
            if (item > 0) {
                if (gesture_index)
                    item++;
                else
                    item--;
                if (item <= 0)
                    item = 4;
                else if (item > 4)
                    item = 1;
                uLCD_display();
            }
            else {
                if (gesture_index)
                    score += 5;
            }
        }
    }
}

void enterSelectionMode(void)
{
    stop = 1;
    audio.spk.pause();
    display = 1;
    item = 1;
    info = 0;
    uLCD_display();
}
void enter(void)
{
    if (item == 1) {
        song++;
        if (song > 2) {
            song = 0;
        }
        loadQ.call(loadSignalHandler);
    }
    else if (item == 2) {
        song--;
        if (song < 0) {
            song = 2;
        }
        loadQ.call(loadSignalHandler);
    }
    else if (item == 3 && !load) {
        display = 2;
        item = 1;
        uLCD_display();
        load = 1;
    }
    else if (item == 3 && load) {
        load = 0;
        loadQ.call(loadSignalHandler);
    }
    else if (item == 4) {
        load = 0;
        loadQ.call(loadSignalHandler);
    }
}
int main(int argc, char* argv[])
{
// start gesture

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor
    //model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32)) {
        //error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }

    if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
    }

    //error_reporter->Report("Set up successful...\n");
// end gesture

    DNN.start(dnn);
    green_led = 1;
    uLCD.color(BLUE);
    uLCD_display();
    MODE.start(callback(&modeQ, &EventQueue::dispatch_forever));
    sw2.fall(modeQ.event(enterSelectionMode));
    // sw3.fall(modeQ.event(exitSelectionMode));

    LOAD.start(callback(&loadQ, &EventQueue::dispatch_forever));
    PLAY.start(callback(&playQ, &EventQueue::dispatch_forever));
    TAIKO.start(callback(&taikoQ, &EventQueue::dispatch_forever));
    load = 0;
    sw3.fall(modeQ.event(enter));
    // sw3.rise(loadQ.event(loadSignalHandler));

}
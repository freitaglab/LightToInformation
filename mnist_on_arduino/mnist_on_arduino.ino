/******************************************************************************
 * File:     mnist_on_arduino.ino
 * Author:   Michael Rinderle
 * Email:    michael.rinderle@tum.de
 * Created:  27.04.2019
 *
 * Revisions: ---
 *
 * Description: This program implements the neural network to predict hand
 *              written digits from 14x14 pixel MNIST images.
 *              The images are received via the serial port and the predicted
 *              value is sent to the serial port.
 *
 ******************************************************************************/


#define STARTMARKER 'S'
#define ENDMARKER 'E'
#define SEPARATOR ','
#define BUFFERSIZE 8
char receivebuffer[BUFFERSIZE]; // array to store received characters
byte buffer_idx = 0;

#include "network.h"            // include weights and biases of neural network

byte image[img_size];           // array to store input image
byte ctr = 0;
bool imageReceived = false;

long layer1[l1_size];           // array to store layer 1 results
long layer2[l2_size];           // array to store layer 2 results

#define l1_shift (l1w_bits + img_bits - l1b_bits)       // bit-shift distance of layer 1
#define l2_shift (l2w_bits + l1b_bits - l2b_bits)       // bit-shift distance of layer 2


void setup() {
    Serial.begin(1000000);   // initialize serial port
    Serial.println("<Arduino is ready>");
}

void loop() {
    receive_image();

    if (imageReceived) {
        compute_network();
        send_result();

        imageReceived = false;
        ctr = 0;
    }
}


/**
 * Function to receive a 14x14 pixel image encoded in 8-bit unsigned integers
 * from the serial port
 */
void receive_image() {
    static bool receiveFlag = false;
    char rc;

    while (Serial.available() > 0 && imageReceived == false) {
        rc = Serial.read();     // read character from serial port

        if (receiveFlag) {
            if (rc == ENDMARKER) {
                // receivebuffer[buffer_idx] = '\n';  // terminate string
                parse_buffer();                       // parse buffer
                receiveFlag = false;
                imageReceived = true;
            } else if (rc == SEPARATOR) {
                // receivebuffer[buffer_idx] = '\n';  // terminate string
                parse_buffer();                       // parse buffer
            } else {
                receivebuffer[buffer_idx] = rc;
                buffer_idx++;

                // TODO: What if buffer_idx > BUFFERSIZE
            }
        } else if (rc == STARTMARKER) {
            receiveFlag = true;   // start receiving data
            buffer_idx = 0;       // reset buffer index
        }
    }
}

void parse_buffer() {
    if (ctr < img_size) {
        image[ctr] = atoi(receivebuffer);
        ctr++;
    } else {
        imageReceived = true;
    }

    // reset buffer
    for (byte i=0; i<BUFFERSIZE; ++i) {
        receivebuffer[i] = '\n';
    }
    buffer_idx = 0;
}


/**
 * Function to compute the 2 layer neural network.
 * Weights and Biases are loaded from PROGMEM
 */
void compute_network() {
    // COMPUTE LAYER 1
    for (byte i=0; i<l1_size; ++i) {
        layer1[i] = 0;

        // multiply matrix row i and vector
        for (byte j=0; j<img_size; ++j) {
            // layer1[i] += (long)image[j] * (long)(char)pgm_read_byte(&l1_weights[(int)j + (int)i * img_size]);
            layer1[i] += (long)image[j] * (long)(char)pgm_read_byte(&l1_weights[i][j]);
        }

        // shift bits to have same precision as layer 1 bias
        layer1[i] = layer1[i] >> l1_shift;

        // add layer 1 bias
        layer1[i] += (long)(char)pgm_read_byte(&l1_bias[i]);

        // relu activation
        if (layer1[i] < 0) {
            layer1[i] = 0;
        }
    }

    // COMPUTE LAYER 2
    for (byte i=0; i<l2_size; ++i) {
        layer2[i] = 0;

        // multiply matrix row i and vector
        for (byte j=0; j<l1_size; ++j) {
            // layer2[i] += layer1[j] * (long)(char)pgm_read_byte(&l2_weights[(int)j + (int)i * l1_size]);
            layer2[i] += layer1[j] * (long)(char)pgm_read_byte(&l2_weights[i][j]);
        }

        // shift bits to have same precision as layer 2 bias
        layer2[i] = layer2[i] >> l2_shift;

        // add layer 2 bias
        layer2[i] += (long)(char)pgm_read_byte(&l2_bias[i]);
    }
}


/**
 * Function to send the predicted value back to the serial port
 */
void send_result() {
    byte max_idx = 0;
    long max_val = 0;

    for (byte i = 0; i < l2_size; ++i) {
        if (layer2[i] > max_val) {
            max_val = layer2[i];
            max_idx = i;
        }
    }

    Serial.println("RESULT");
    Serial.println(max_idx);
}

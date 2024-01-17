#include <zephyr/kernel.h>
#include <math.h>
#include "confusion.h"
#include "adc.h"
#include "neural_network.h"

// Global variables for weights and biases
double weights_layer_0[LAYER_0_NEURONS * INPUT_DATA_SIZE];
double biases_layer_0[LAYER_0_NEURONS];
double weights_layer_2[LAYER_2_NEURONS * LAYER_1_NEURONS];
double biases_layer_2[LAYER_2_NEURONS];

void initializeNeuralNetwork(void)
{
    // Example of initializing weights and biases for Layer 0
    // These values need to be replaced with your actual trained weights and biases
    double weights_0_values[] = {
        0.3036405146121979, 0.24570688605308533, 0.145527184009552, -0.45107346773147583,
        -0.20215854048728943, -0.04948733374476433, -0.029027696698904037, -0.45735663175582886,
        0.3682192265987396, 0.07978120446205139, -0.20779955387115479, -0.38776350021362305,
        -0.39982593059539795, -0.1261535882949829, -0.19824664294719696, 0.06371012330055237,
        -0.35693883895874023, -0.34639430046081543, 0.2010888308286667, -0.42040136456489563,
        0.104937344789505, -0.13197436928749084, -0.08856919407844543, -0.38764411211013794,
        0.059484731405973434, 0.1955423653125763, -0.26626044511795044, -0.023263877257704735,
        0.27822211384773254, 0.17185768485069275, -0.35959452390670776, -0.3598894476890564,
        -0.2824243903160095, -0.1250041425228119, 0.3915944993495941, 0.06559625267982483,
        0.03830006718635559, -0.28623056411743164, -0.4329483211040497, -0.14602723717689514,
        -0.0009583384962752461, 0.38147351145744324, 0.3470211625099182, -0.4701763689517975,
        -0.4709697961807251, -0.32061636447906494, -0.21941912174224854, 0.07032915949821472,
        -0.20024406909942627, 0.07351060956716537, -0.29237422347068787, 0.22572527825832367,
        -0.24106347560882568, -0.08797353506088257, 0.35247424244880676, -0.007615178823471069,
        -0.08838523179292679, -0.19637644290924072, 0.11068207025527954, 0.1363341510295868,
        -0.1658509075641632, 0.015379160642623901, -0.2037406712770462, -0.14090460538864136,
        -0.5206899046897888, -0.36063021421432495};

    double biases_0_values[] = {
        0.0, 0.0, -0.10526704788208008, 0.0, 0.0, -0.25725141167640686,
        -0.1598031222820282, 0.016801742836833, 0.0, 0.0, 0.0, 0.0,
        -0.03994191437959671, 0.0, -0.07373754680156708, 0.0, 0.0, 0.0,
        -0.12127702683210373, 0.0, -0.06894215196371078, 0.0};

    for (int i = 0; i < LAYER_0_NEURONS * INPUT_DATA_SIZE; i++)
    {
        weights_layer_0[i] = weights_0_values[i];
    }
    for (int i = 0; i < LAYER_0_NEURONS; i++)
    {
        biases_layer_0[i] = biases_0_values[i];
    }

    // Similar initialization for Layer 2
    double weights_2_values[] = {
        -0.4424046576023102, 0.44084250926971436, 0.07702809572219849, -0.43659210205078125,
        -0.053778767585754395, -0.04353964328765869, -0.44143882393836975, 0.11249411106109619,
        0.029383838176727295, -0.26440566778182983, 0.08480340242385864, 0.02446877956390381, 0.24324822425842285,
        0.3116787075996399, -0.023866135627031326, -0.24606290459632874, 0.31228962540626526, -0.3075824975967407,
        -0.16022634506225586, -0.14252004027366638, 0.11694884300231934, 0.20088410377502441, -0.27686989307403564,
        -0.35739749670028687, -0.38767462968826294, 0.2953444719314575, 0.44506126642227173, 0.19526809453964233,
        0.3271276354789734, -0.16063722968101501, -0.05016192048788071, -0.18065793812274933, 0.08678044378757477,
        -0.15127032995224, -0.18875980377197266, 0.10685499012470245, 0.0012856441317126155, -0.18572410941123962,
        -0.3292396068572998, 0.38250258564949036, -0.11287606507539749, -0.19856831431388855, 0.4633263647556305,
        -0.1761886328458786, -0.25093457102775574, -0.4350983202457428, -0.14724372327327728, -0.15742874145507812,
        -0.1954023241996765, 0.2998465299606323, 0.3649037480354309, -0.3735803961753845, -0.10821762681007385, 0.18028199672698975,
        -0.12077754735946655, 0.1077035665512085, 0.21127408742904663, -0.08662071824073792, -0.28928279876708984, 0.3451104164123535,
        -0.15302449464797974, -0.13402891159057617, 0.05836987495422363, -0.14853248000144958, 0.22795706987380981, -0.3153206706047058,
         0.3052860498428345, -0.3888515830039978, 0.11564505100250244, -0.36346864700317383, 0.06829208135604858, 0.006894916296005249, 
         0.2612857222557068, 0.034804556518793106, 0.32042160630226135, 0.11562849581241608, 0.36020293831825256, 0.3468048572540283, 0.2969437837600708, 
         0.27903950214385986, 0.34419578313827515, 0.10937082767486572, 0.22033685445785522, 0.17138713598251343, -0.15806743502616882, 0.22218191623687744, 
         0.15800312161445618, 0.0765981674194336, 0.11728585511445999, 0.4133191406726837, -0.09842029213905334, 0.3442125916481018, -0.052541881799697876,
          0.17334413528442383, -0.08334767818450928, -0.28429436683654785, 0.11638796329498291, 0.09368008375167847, 0.1503257155418396, 0.23873579502105713,
           0.28531956672668457, 0.07805430889129639, -0.10158571600914001, -0.4326690137386322, 0.416534423828125, 0.027358591556549072, 0.02128702402114868,
            0.12928426265716553, -0.27185097336769104, 0.1537780910730362, -0.3909333348274231, -0.16267596185207367, 0.12906098365783691, -0.34799209237098694,
             0.2958993911743164, 0.21113282442092896, 0.3876293897628784, -0.25001323223114014, -0.11623552441596985, 0.30440694093704224, 0.0928691178560257, 
             -0.05533836781978607, 0.39085233211517334, 0.10745985060930252, 0.28592175245285034, 0.1778510957956314, -0.30951637029647827, -0.08690589666366577,
              0.1902216076850891, 0.0532151460647583, 0.1350029706954956, 0.23407769203186035};

    double biases_2_values[] = {
        -0.022157272323966026, 0.007897702977061272, 0.07932887226343155, 0.10266172140836716, -0.14710038900375366, -0.02624308317899704};

    for (int i = 0; i < LAYER_2_NEURONS * LAYER_1_NEURONS; i++)
    {
        weights_layer_2[i] = weights_2_values[i];
    }
    for (int i = 0; i < LAYER_2_NEURONS; i++)
    {
        biases_layer_2[i] = biases_2_values[i];
    }
}

int predictClass(double x, double y, double z)
{
    double input[INPUT_DATA_SIZE] = {x, y, z};
    double predictions[LAYER_2_NEURONS];
    forward_pass(input, predictions, weights_layer_0, biases_layer_0, weights_layer_2, biases_layer_2);
    return get_predicted_class(predictions, LAYER_2_NEURONS);
}

double calculate_activation(double neuron_weights[], double neuron_bias, double input_data[], int input_size)
{
    double activation = neuron_bias;
    for (int i = 0; i < input_size; i++)
    {
        activation += neuron_weights[i] * input_data[i];
    }
    return activation;
}

double relu(double activation)
{
    return activation > 0 ? activation : 0;
}

void softmax(double final_output[], int size)
{
    double max_value = final_output[0];
    for (int i = 1; i < size; i++)
    {
        if (final_output[i] > max_value)
        {
            max_value = final_output[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        final_output[i] = exp(final_output[i] - max_value);
        sum += final_output[i];
    }
    for (int i = 0; i < size; i++)
    {
        final_output[i] /= sum;
    }
}

void forward_pass(double input[], double predictions[], double weights_layer_0[], double biases_layer_0[], double weights_layer_2[], double biases_layer_2[])
{
    double layer_1_output[LAYER_1_NEURONS];
    for (int i = 0; i < LAYER_1_NEURONS; i++)
    {
        double activation = calculate_activation(weights_layer_0 + i * INPUT_DATA_SIZE, biases_layer_0[i], input, INPUT_DATA_SIZE);
        layer_1_output[i] = relu(activation);
    }

    double final_output[LAYER_2_NEURONS];
    for (int i = 0; i < LAYER_2_NEURONS; i++)
    {
        double activation = calculate_activation(weights_layer_2 + i * LAYER_1_NEURONS, biases_layer_2[i], layer_1_output, LAYER_1_NEURONS);
        final_output[i] = activation;
    }

    softmax(final_output, LAYER_2_NEURONS);
    for (int i = 0; i < LAYER_2_NEURONS; i++)
    {
        predictions[i] = final_output[i];
    }
}

int get_predicted_class(double predictions[], int size)
{
    int predicted_class = 0;
    double max_prob = predictions[0];
    for (int i = 1; i < size; i++)
    {
        if (predictions[i] > max_prob)
        {
            max_prob = predictions[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

int CP[6][3] = {
    {1320.444444, 1630.296296, 1629.148148},
    {1969.857143, 1602.607143, 1620.428571},
    {1623.035714, 1282.500000, 1609.678571},
    {1664.851852, 1948.481481, 1642.592593},
    {1640.785714, 1633.642857, 1312.321429},
    {1644.892857, 1620.178571, 1956.250000}};

int measurements[6][3] = {
    {1320.444444, 1630.296296, 1629.148148},
    {1969.857143, 1602.607143, 1620.428571},
    {1623.035714, 1282.500000, 1609.678571},
    {1664.851852, 1948.481481, 1642.592593},
    {1640.785714, 1633.642857, 1312.321429},
    {1644.892857, 1620.178571, 1956.250000}};

int CM[6][6] = {0};

void printConfusionMatrix(void)
{
    printk("Confusion matrix = \n");
    printk("   cp1 cp2 cp3 cp4 cp5 cp6\n");
    for (int i = 0; i < 6; i++)
    {
        printk("cp%d %d   %d   %d   %d   %d   %d\n", i + 1, CM[i][0], CM[i][1], CM[i][2], CM[i][3], CM[i][4], CM[i][5]);
    }
}


void makeOneClassificationAndUpdateConfusionMatrix(int direction) {
    for (int i = 0; i < 100; i++) {
        struct Measurement m = readADCValue();
        printk("x: %d, y: %d, z: %d\n", m.x, m.y, m.z);
        int predictedClass = predictClass(m.x, m.y, m.z);
        if (predictedClass >= 0) {
            CM[direction][predictedClass]++;
        }
    }
    printPerformanceMetrics(CM);
}

void makeHundredFakeClassifications(void)
{
    for (int i = 0; i < 100; i++)
    {
        int randomIndex = rand() % 6;
        makeOneClassificationAndUpdateConfusionMatrix(randomIndex);
    }
}

void printPerformanceMetrics(int CM[6][6]) {
    int totalPredictions = 0;
    int correctPredictions = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                correctPredictions += CM[i][j];
            }
            totalPredictions += CM[i][j];
        }
    }

    double accuracy = (double)correctPredictions / totalPredictions;
    printk("Total Predictions: %d, Correct Predictions: %d\n", totalPredictions, correctPredictions);
    printk("Accuracy: %lf\n", accuracy);
}

void resetConfusionMatrix(void) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            CM[i][j] = 0;
        }
    }
}

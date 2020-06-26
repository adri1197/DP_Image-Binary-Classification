import os
import sys
from datetime import datetime

from .classification import Classification
from .parameters import parse_args
from .utils.constants import MULTI_DEVICE_NAME
from .utils.inputs_filling import set_inputs
from .utils.logging import logger
from .utils.progress_bar import ProgressBar
from .utils.show_results import ShowResults
from .utils.utils import next_step, config_network_inputs, get_number_iterations, \
    process_help_inference_string, print_perf_counters, get_duration_in_milliseconds, \
    get_command_line_arguments
from .utils.statistics_report import StatisticsReport, averageCntReport, detailedCntReport


def main():
    # ------------------------------ 1. Parsing and validating input arguments -------------------------------------
    next_step()
    run(parse_args())

def run(args):
    statistics = None
    try:
        if args.number_streams is None:
                logger.warn(" -nstreams default value is determined automatically for a device. "
                            "Although the automatic selection usually provides a reasonable performance, "
                            "but it still may be non-optimal for some cases, for more information look at README. ")

        if args.report_type:
          statistics = StatisticsReport(StatisticsReport.Config(args.report_type, args.report_folder))
          statistics.add_parameters(StatisticsReport.Category.COMMAND_LINE_PARAMETERS, get_command_line_arguments(sys.argv))


        # ------------------------------ 2. Loading Inference Engine ---------------------------------------------------
        next_step(step_id=2)

        device_name = args.target_device.upper()

        classification = Classification(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type)

        classification.add_extension(args.path_to_extension, args.path_to_cldnn_config)

        version = classification.get_version_info()

        logger.info(version)

        # --------------------- 3. Read the Intermediate Representation of the network ---------------------------------
        next_step()

        start_time = datetime.utcnow()
        ie_network = classification.read_network(args.path_to_model)
        duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
        logger.info("Read network took {} ms".format(duration_ms))
        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('read network time (ms)', duration_ms)
                                      ])

        # --------------------- 4. Resizing network to match image sizes and given batch -------------------------------

        next_step()
        if args.batch_size and args.batch_size != ie_network.batch_size:
            classification.reshape(ie_network, args.batch_size)
        batch_size = ie_network.batch_size
        logger.info('Network batch size: {}'.format(ie_network.batch_size))

        # --------------------- 5. Configuring input of the model ------------------------------------------------------
        next_step()

        config_network_inputs(ie_network)

        # --------------------- 6. Setting device configuration --------------------------------------------------------
        next_step()
        classification.set_config(args.number_streams, args.api_type, args.number_threads,
                             args.infer_threads_pinning)

        # --------------------- 7. Loading the model to the device -----------------------------------------------------
        next_step()

        start_time = datetime.utcnow()
        perf_counts = True if args.perf_counts or \
                              args.report_type in [ averageCntReport, detailedCntReport ] else False
        exe_network = classification.load_network(ie_network, perf_counts)
        duration_ms = "{:.2f}".format((datetime.utcnow() - start_time).total_seconds() * 1000)
        logger.info("Load network took {} ms".format(duration_ms))
        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('load network time (ms)', duration_ms)
                                      ])

        # --------------------- 8. Setting optimal runtime parameters --------------------------------------------------
        next_step()

        # Number of requests
        infer_requests = exe_network.requests

        # Iteration limit

        classification.niter = get_number_iterations(classification.niter, classification.nireq, args.api_type)

        # ------------------------------------ 9. Creating infer requests and filling input blobs ----------------------
        next_step()

        paths_to_input = list()
        if args.paths_to_input:
            for path in args.paths_to_input:
                paths_to_input.append(os.path.abspath(*path) if args.paths_to_input else None)

        images_path = set_inputs(paths_to_input, batch_size, ie_network.inputs, infer_requests)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                      [
                                          ('topology', ie_network.name),
                                          ('target device', device_name),
                                          ('API', args.api_type),
                                          ('precision', str(ie_network.precision)),
                                          ('batch size', str(batch_size)),
                                          ('number of iterations', str(classification.niter) if classification.niter else "0"),
                                          ('number of parallel infer requests', str(classification.nireq)),
                                          ('duration (ms)', str(get_duration_in_milliseconds(classification.duration_seconds))),
                                       ])

            for nstreams in classification.device_number_streams.items():
                statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                         [
                                            ("number of {} streams".format(nstreams[0]), str(nstreams[1])),
                                         ])

        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(classification)

        next_step(additional_info=output_string)
        progress_bar_total_count = 10000
        if classification.niter and not classification.duration_seconds:
            progress_bar_total_count = classification.niter

        progress_bar = ProgressBar(progress_bar_total_count, args.stream_output, args.progress) if args.progress else None

        output_inference,fps, latency_ms, total_duration_sec, iteration = classification.infer(exe_network, batch_size, progress_bar)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

        if perf_counts:
            perfs_count_list = []
            for ni in range(int(classification.nireq)):
                perfs_count_list.append(exe_network.requests[ni].get_perf_counts())
            if args.perf_counts:
                print_perf_counters(perfs_count_list)
            if statistics:
              statistics.dump_performance_counters(perfs_count_list)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('total execution time (ms)', '{:.2f}'.format(get_duration_in_milliseconds(total_duration_sec))),
                                          ('total number of iterations', str(iteration)),
                                      ])
            if MULTI_DEVICE_NAME not in device_name:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('latency (ms)', '{:.2f}'.format(latency_ms)),
                                          ])

            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('throughput', '{:.2f}'.format(fps)),
                                      ])

        if statistics:
          statistics.dump()

        print('Count:      {} iterations'.format(iteration))
        print('Duration:   {:.2f} ms'.format(get_duration_in_milliseconds(total_duration_sec)))
        if MULTI_DEVICE_NAME not in device_name:
            print('Latency:    {:.2f} ms'.format(latency_ms))
        print('Throughput: {:.2f} FPS'.format(fps))
        

        del exe_network

        # ------------------------------------ 12. Showing the results of the inferences -------------------------------------------
        next_step()
        labels = ['damaged','undamaged']
        ShowResults.showOutput(args.api_type,batch_size,output_inference,images_path,labels)


        next_step.step_id = 0
    except Exception as e:
        logger.exception(e)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('error', str(e)),
                                      ])
            statistics.dump()
        sys.exit(1)

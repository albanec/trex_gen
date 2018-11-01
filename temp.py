def _trex_run (job_summary, m, duration):
    trex_thread = job_summary['trex_thread']

    p = ProgressBar(duration, trex_thread.get_router())
    p.start()

    try:
        results = trex_thread.run(m, duration)
    except Exception as e:
        p.stop()
        raise

    p.stop()

    if (results == None):
        raise Exception("Failed to run Trex")

    # fetch values
    trex_r = results['trex_results']
    avc_r  = results['avc_results']

    sanity_test_run(trex_r, avc_r)

    res_dict = {}

    res_dict['m']  = m
    total_tx_bps = trex_r.get_last_value("trex-global.data.m_tx_bps")
    res_dict['tx'] = total_tx_bps / (1000 * 1000)  # EVENTUALLY CONTAINS IN MBPS (EXTRACTED IN BPS)

    res_dict['cpu_util'] = avc_r['cpu_util']

    if int(res_dict['cpu_util']) == 0:
        res_dict['norm_cpu']=1;
    else:
        res_dict['norm_cpu'] = (res_dict['tx'] / res_dict['cpu_util']) * 100

    res_dict['maximum-latency']  = max ( trex_r.get_max_latency().values() ) #trex_r.res['maximum-latency']
    res_dict['average-latency']  = trex_r.get_avg_latency()['all'] #trex_r.res['average-latency']
    
    logger.log(cpu_histo_to_str(avc_r['cpu_histo']))

    res_dict['total-pkt-drop'] = trex_r.get_total_drops()
    res_dict['expected-bps'] = trex_r.get_expected_tx_rate()['m_tx_expected_bps']
    res_dict['total-pps'] = get_median( trex_r.get_value_list("trex-global.data.m_tx_pps") )#trex_r.res['total-pps']
    res_dict['m_total_pkt'] = trex_r.get_last_value("trex-global.data.m_total_tx_pkts")

    res_dict['latency_condition'] = job_summary['trex_params']['trex_latency_condition']

    return res_dict

def trex_run (job_summary, m, duration):
    res = _trex_run (job_summary, m, duration)
    return res





######################## describe a find job ########################
class FindJob:
    # init a job object with min / max
    def __init__ (self, min, max, job_summary):
        self.min = float(min)
        self.max = float(max)
        self.job_summary = job_summary
        self.cond_type = job_summary['cond_type']
        self.success_points = []
        self.iter_num = 1
        self.found = False
        self.iter_duration = 300

    def _distance (self):
        return ( (self.max - self.min) / min(self.max, self.min) )

    def time_to_end (self):
        time_in_sec = (self.iters_to_end() * self.iter_duration)
        return timedelta(seconds = time_in_sec)

    def iters_to_end (self):
        # find 2% point
        ma = self.max
        mi = self.min
        iter = 0

        while True:
            dist = (ma - mi) / min(ma , mi)
            if dist < 0.02:
                break
            if random.choice(["up", "down"]) == "down":
                ma = (ma + mi) / 2
            else:
                mi = (ma + mi) / 2

            iter += 1

        return (iter)

    def _cur (self):
        return ( (self.min + self.max) / 2 )

    def _add_success_point (self, res_dict):
        self.success_points.append(res_dict.copy())

    def _is_found (self):
        return (self.found)

    def _next_iter_duration (self):
        return (self.iter_duration)

    # execute iteration
    def _execute (self):
        # reset the found var before running
        self.found = False

        # run and print results
        res_dict = trex_run(self.job_summary, self._cur(), self.iter_duration)

        self.iter_num += 1
        cur = self._cur()

        if (self._distance() < 0.02):
            if (check_condition(self.cond_type, res_dict)):
                # distance < 2% and success - we are done
                self.found = True
            else:
                # lower to 90% of current and retry
                self.min = cur * 0.9
                self.max = cur
        else:
            # success
            if (check_condition(self.cond_type, res_dict)):
                self.min = cur
            else:
                self.max = cur

        if (check_condition(self.cond_type, res_dict)):
            self._add_success_point(res_dict)

        return res_dict

    # find the max M before 
    def find_max_m (self):

        res_dict = {}
        while not self._is_found():

            logger.log("\n-> Starting Find Iteration #{0}\n".format(self.iter_num))
            logger.log("Estimated BW         ~=  {0:,.2f} [Mbps]".format(m_to_mbps(self.job_summary, self._cur())))
            logger.log("M                     =  {0:.6f}".format(self._cur()))
            logger.log("Duration              =  {0} seconds".format(self._next_iter_duration()))
            logger.log("Current BW Range      =  {0:,.2f} [Mbps] / {1:,.2f} [Mbps]".format(m_to_mbps(self.job_summary, self.min), m_to_mbps(self.job_summary, self.max)))
            logger.log("Est. Iterations Left  =  {0} Iterations".format(self.iters_to_end()))
            logger.log("Est. Time Left        =  {0}\n".format(self.time_to_end()))

            res_dict = self._execute()

            print_trex_results(res_dict, self.cond_type)

        find_results = res_dict.copy()
        find_results['max_m'] = self._cur()
        return (find_results)








# find the correct range of M
def find_m_range (job_summary):
    trex = job_summary['trex']
    trex_config = job_summary['trex_params']

    # if not provided - guess the correct range of bandwidth
    if not job_summary['m_range']:
        m_range = [0.0, 0.0]
        # 1 Mbps -> 1 Gbps
        LOW_TX = 1.0 * 1000 * 1000
        MAX_TX = 1.0 * 1000 * 1000 * 1000

        # for 10g go to 10g
        if trex_config['trex_machine_type'] == "10G":
            MAX_TX *= 10
   
        # dual injection can potentially reach X2 speed
        if trex_config['trex_is_dual'] == True:
            MAX_TX *= 2
1
    else:
        m_range = job_summary['m_range']
        LOW_TX = m_range[0] * 1000 * 1000
        MAX_TX = m_range[1] * 1000 * 1000
   
   
    logger.log("\nSystem Settings - Min: {0:,} Mbps / Max: {1:,} Mbps".format(LOW_TX / (1000 * 1000), MAX_TX / (1000 * 1000)))
    logger.log("\nTrying to get system minimum M and maximum M...")
   
    res_dict = trex_run(job_summary, 1, 30)
   
    # figure out low / high M
    m_range[0] = (LOW_TX / res_dict['expected-bps']) * 1
    m_range[1] = (MAX_TX / res_dict['expected-bps']) * 1


    # return both the m_range and the base m unit for future calculation
    results = {}
    results['m_range'] = m_range
    results['base_m_unit'] = res_dict['expected-bps'] /(1000 * 1000)

    return (results)

#
# Load config params
#
def load_trex_config_params (filename, yaml_file):
    config = {}

    parser = ConfigParser.ConfigParser()

    try:
        parser.read(filename)

        config['trex_name'] = parser.get("trex", "machine_name")
        config['trex_port'] = parser.get("trex", "machine_port")
        config['trex_hisory_size'] = parser.getint("trex", "history_size")

        config['trex_latency_condition'] = parser.getint("trex", "latency_condition")
        config['trex_yaml_file'] = yaml_file

        # support legacy data
        config['trex_latency'] = parser.getint("trex", "latency")
        config['trex_limit_ports'] = parser.getint("trex", "limit_ports")
        config['trex_cores'] = parser.getint("trex", "cores")
        config['trex_machine_type'] = parser.get("trex", "machine_type")
        config['trex_is_dual'] = parser.getboolean("trex", "is_dual")

        # optional Trex parameters
        if parser.has_option("trex", "config_file"):
            config['trex_config_file'] = parser.get("trex", "config_file")
        else:
            config['trex_config_file'] = None

        if parser.has_option("trex", "misc_params"):
            config['trex_misc_params'] = parser.get("trex", "misc_params")
        else:
            config['trex_misc_params'] = None

        # router section
        
        if parser.has_option("router", "port"):
            config['router_port'] = parser.get("router", "port")
        else:
            # simple telnet port
            config['router_port'] = 23

        config['router_interface'] = parser.get("router", "interface")
        config['router_password'] = parser.get("router", "password")
        config['router_type'] = parser.get("router", "type")

    except Exception as inst:
        raise TrexRunException("\nBad configuration file: '{0}'\n\n{1}".format(filename, inst))

    return config

#
# Prepare for run fun
#
def prepare_for_run (job_summary):
    global logger
    
    # generate unique id
    job_summary['job_id'] = generate_job_id()
    job_summary['job_dir'] = "trex_job_{0}".format(job_summary['job_id'])
    
    job_summary['start_time'] = datetime.datetime.now()

    if not job_summary['email']:
        job_summary['user'] = getpass.getuser() 
        job_summary['email'] = "{0}@cisco.com".format(job_summary['user'])

    # create dir for reports
    try:
        job_summary['job_dir'] = os.path.abspath( os.path.join(os.getcwd(), 'logs', job_summary['job_dir']) )
        print(job_summary['job_dir'])
        os.makedirs( job_summary['job_dir'] )
        
    except OSError as err:
        if err.errno == errno.EACCES:
            # fall back. try creating the dir name at /tmp path
            job_summary['job_dir'] = os.path.join("/tmp/", "trex_job_{0}".format(job_summary['job_id']) )
            os.makedirs(job_summary['job_dir'])

    job_summary['log_filename'] = os.path.join(job_summary['job_dir'], "trex_log_{0}.txt".format(job_summary['job_id']))
    job_summary['graph_filename'] = os.path.join(job_summary['job_dir'], "trex_graph_{0}.html".format(job_summary['job_id']))

    # init logger
    logger = MyLogger(job_summary['log_filename'])

    # mark those as not populated yet
    job_summary['find_results'] = None
    job_summary['plot_results'] = None

    # create trex client instance
    trex_params = load_trex_config_params(job_summary['config_file'],job_summary['yaml'])
    trex = CTRexClient(trex_host = trex_params['trex_name'], 
        max_history_size = trex_params['trex_hisory_size'], 
        trex_daemon_port = trex_params['trex_port'])

    job_summary['trex'] = trex
    job_summary['trex_params'] = trex_params

    # create trex task thread
    job_summary['trex_thread'] = CTRexWithRouter(trex, trex_params);

    # in case of an error we need to call the remote cleanup
    cleanup_list.append(trex.stop_trex)
    
    # signal handler
    signal.signal(signal.SIGINT, int_signal_handler)
    signal.signal(signal.SIGUSR1, error_signal_handler)

def after_run (job_summary):
    job_summary['total_run_time'] = datetime.datetime.now() - job_summary['start_time']
    reporter = JobReporter(job_summary)
    reporter.print_summary()
    reporter.send_email_report()

def launch (job_summary):
    prepare_for_run(job_summary)

    print_header()
  
    log_startup_summary(job_summary)

    # find the correct M range if not provided
    range_results = find_m_range(job_summary)
    
    job_summary['base_m_unit'] = range_results['base_m_unit']

    if job_summary['m_range']:
        m_range = job_summary['m_range']
    else:
        m_range = range_results['m_range']

    logger.log("\nJob Bandwidth Working Range:\n")
    logger.log("Min M = {0:.6f} / {1:,.2f} [Mbps] \nMax M = {2:.6f} / {3:,.2f} [Mbps]".format(m_range[0], m_to_mbps(job_summary, m_range[0]), m_range[1], m_to_mbps(job_summary, m_range[1])))

    # job time
    findjob = FindJob(m_range[0], m_range[1], job_summary)
    job_summary['find_results'] = findjob.find_max_m()

    if job_summary['job_type'] == "all":
        # plot points to graph
        plotjob = PlotJob(findjob)
        job_summary['plot_results'] = plotjob.plot()

    after_run(job_summary)

# populate the fields for run
def populate_fields (job_summary, args):
    job_summary['config_file'] = args.config_file
    job_summary['job_type'] = args.job
    job_summary['cond_type'] = args.cond_type
    job_summary['yaml'] = args.yaml

    if args.n:
        job_summary['job_name'] = args.n
    else:
        job_summary['job_name'] = "Nameless"

    # did the user provided an email
    if args.e:
        job_summary['email'] = args.e
    else:
        job_summary['email'] = None

    # did the user provide a range ?
    if args.m:
        job_summary['m_range'] = args.m
    else:
        job_summary['m_range'] = None

    # some pretty shows
    job_summary['cond_name'] = 'Drop Pkt' if (args.cond_type == 'drop') else 'High Latency'

    if args.job == "find":
        job_summary['job_type_str'] = "Find Max BW"
    elif args.job == "plot":
        job_summary['job_type_str'] = "Plot Graph"
    else:
        job_summary['job_type_str'] = "Find Max BW & Plot Graph"

    if args.job != "find":
        verify_glibc_version()
        

def entry ():

    job_summary = {}

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="Job Name",
                        type = str)

    parser.add_argument("-m", help="M Range [default: auto calcuation]",
                        nargs = 2,
                        type = float)

    parser.add_argument("-e", help="E-Mail for report [default: whoami@cisco.com]",
                        type = str)

    parser.add_argument("-c", "--cfg", dest = "config_file", required = True, 
                        help = "Configuration File For Trex/Router Pair",
                        type = lambda x: is_valid_file(parser, "config file does not exists",x))

    parser.add_argument("job", help = "Job type",
                        type = str,
                        choices = ['find', 'plot', 'all'])

    parser.add_argument("cond_type", help="type of failure condition",
                        type = str,
                        choices = ['latency','drop'])

    parser.add_argument("-f", "--yaml", dest = "yaml", required = True,
                        help="YAML file to use", type = str)

    args = parser.parse_args()

    with TermMng():
        try:
            populate_fields(job_summary, args)
            launch(job_summary)

        except Exception as e:
            ErrorHandler(e, traceback.format_exc())

    logger.log("\nReport bugs to imarom@cisco.com\n")
    g_stop = True


def dummy_test ():
    job_summary = {}
    find_results = {}

    job_summary['config_file'] = 'config/trex01-1g.cfg'
    job_summary['yaml'] = 'dummy.yaml'
    job_summary['email'] = 'imarom@cisco.com'
    job_summary['job_name'] = 'test'
    job_summary['job_type_str'] = 'test'

    prepare_for_run(job_summary)

    time.sleep(2)
    job_summary['yaml'] = 'dummy.yaml'
    job_summary['job_type']  = 'find'
    job_summary['cond_name'] = 'Drop'
    job_summary['cond_type'] = 'drop'
    job_summary['job_id']= 94817231
    
    find_results['tx'] = 210.23
    find_results['m'] = 1.292812
    find_results['total-pps'] = 1000
    find_results['cpu_util'] = 74.0
    find_results['maximum-latency'] = 4892
    find_results['average-latency'] = 201
    find_results['total-pkt-drop'] = 0

    findjob = FindJob(1,1,job_summary)
    plotjob = PlotJob(findjob)
    job_summary['plot_results'] = plotjob.plot()

    job_summary['find_results'] = find_results
    job_summary['plot_results'] = [{'cpu_util': 2.0,'norm_cpu': 1.0,  'total-pps': 1000, 'expected-bps': 999980.0, 'average-latency': 85.0, 'tx': 0.00207*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 221.0},
                                   {'cpu_util': 8.0,'norm_cpu': 1.0,  'total-pps': 1000,'expected-bps': 48500000.0, 'average-latency': 87.0, 'tx': 0.05005*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 279.0},
                                   {'cpu_util': 14.0,'norm_cpu': 1.0, 'total-pps': 1000,'expected-bps': 95990000.0, 'average-latency': 92.0, 'tx': 0.09806*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 273.0},
                                   {'cpu_util': 20.0,'norm_cpu': 1.0, 'total-pps': 1000,'expected-bps': 143490000.0, 'average-latency': 95.0, 'tx': 0.14613*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 271.0},
                                   {'cpu_util': 25.0,'norm_cpu': 1.0, 'total-pps': 1000,'expected-bps': 190980000.0, 'average-latency': 97.0, 'tx': 0.1933*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 302.0},
                                   {'cpu_util': 31.0,'norm_cpu': 1.0, 'total-pps': 1000,'expected-bps': 238480000.0, 'average-latency': 98.0, 'tx': 0.24213*1000, 'total-pkt-drop': 1.0, 'maximum-latency': 292.0},
                                   {'cpu_util': 37.0,'norm_cpu': 1.0, 'total-pps': 1000, 'expected-bps': 285970000.0, 'average-latency': 99.0, 'tx': 0.29011*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 344.0},
                                   {'cpu_util': 43.0,'norm_cpu': 1.0, 'total-pps': 1000, 'expected-bps': 333470000.0, 'average-latency': 100.0, 'tx': 0.3382*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 351.0},
                                   {'cpu_util': 48.0,'norm_cpu': 1.0, 'total-pps': 1000, 'expected-bps': 380970000.0, 'average-latency': 100.0, 'tx': 0.38595*1000, 'total-pkt-drop': 0.0, 'maximum-latency': 342.0},
                                   {'cpu_util': 54.0,'norm_cpu': 1.0, 'total-pps': 1000, 'expected-bps': 428460000.0, 'average-latency': 19852.0, 'tx': 0.43438*1000, 'total-pkt-drop': 1826229.0, 'maximum-latency': 25344.0}]
    after_run(job_summary)

from contextlib import contextmanager
from time import time

from debug.logger import Logger

debug_layer = -1
time_analysis = {}

@contextmanager
def debug(name, show=False):
    '''这个函数以上下文管理器的方式，为一个代码块命名，如果代码块正常运行，记录代码块的执行时间，如果发生异常，报告异常类型，我们可以通过代码块名称找到问题来源'''
    global debug_layer
    global time_analysis

    print('begin', time_analysis, debug_layer)

    logger = Logger(logname = 'log.txt', loglevel=1 , logger='mine').getlog()

    debug_layer += 1
    start = time()
    spacer = '    ' * debug_layer
    if show:
        print('\n%s------------\n%s%s 开始\n%s------------'%(spacer, spacer, name, spacer))
    try:
        yield
    except Exception as e:
        logger.info('%s: %s代码块以下被exception所结束: %s'%(name, spacer, e))
        debug_layer -= 1
    else:
        if show:
            print('%s------------\n%s%s 正常结束，耗时%f seconds\n%s------------'%(spacer, spacer, name, time()-start, spacer))
        debug_layer -= 1
        if name not in time_analysis.keys():
            time_analysis[name] = time()-start
        else:
            time_analysis[name] += time() - start

        #当最外层debug层结束之后，输出时间占用报告
        if debug_layer == -1:
            time_line = "{0:<40}{1:<40}"
            logger.info('------------\n时间占用报告 \n------------')
            logger.info(time_line.format('内容','耗时', chr(12288)))
            for name, time_consumed in time_analysis.items():
                logger.info(time_line.format(name, time_consumed, chr(12288)))
            time_analysis = {}
            return

def my_print(string):
    print(string)
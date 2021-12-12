from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir)

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_metrics(self, summary, step):
        print("\n----Training step {} summary----".format(step))
        for k, v in summary.items():
            print("{:<40} {:<.2f}".format(k, float(v.result())))
            self._writer.add_scalar(k, float(v.result()), step)
            v.reset_states()
        self._writer.flush()

    # (N, T, C, H, W)
    def log_video(self, images, step=None, name='policy', fps=30):
        self._writer.add_video(name, images, step, fps=fps)
        self._writer.flush()

    def log_images(self, images, step=None, name='policy'):
        self._writer.add_images(name, images, step, dataformats='NHWC')
        self._writer.flush()

    def log_figure(self, figure, step=None, name='policy'):
        self._writer.add_figure(name, figure, step)
        self._writer.flush()

import sys

from redis import Redis
from rq import Queue

from config import get_settings
from recovery import RecoveringWorker


def main() -> None:
    settings = get_settings()
    requested = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else ["queue_real_to_render"]
    )
    names = [f"high_{name}" for name in requested] + requested
    connection = Redis.from_url(
        settings.redis_url,
        health_check_interval=20,
        socket_connect_timeout=5,
        socket_timeout=20,
    )
    queues = [Queue(name, connection=connection) for name in names]
    print(f"[*] Stage-1 worker listening on: {names}")
    worker = RecoveringWorker(
        queues,
        connection=connection,
        maintenance_interval=settings.recovery_interval,
        orphan_grace_seconds=settings.orphan_grace,
    )
    worker.recover_interrupted_jobs()
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()

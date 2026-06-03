from SoccerNet.Downloader import SoccerNetDownloader
import os

download_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'SoccerNet')

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=download_path)
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
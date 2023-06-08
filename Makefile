build:
	sudo docker build -t mockup_streamer .
	
run:
	sudo docker run -ip 127.0.0.1:8080:8080 --rm --name my_mockup_streamer --mount type=bind,source="/home/md/workspace/data/bbciRaw/my_eeg_session/folder_with_vhdr_files",target=/var/eeg,readonly -t mockup_streamer


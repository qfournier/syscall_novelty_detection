# Trace Collection

1. Create a script in the home directory of the web server with the following code:
    ```bash
    echo "Enter the destination folder: "
    read output_folder

    echo "Enter the duration of the trace (seconds): "
    read duration

    # Create the session
    lttng create my-session --output=$output_folder

    # Enable user-space events used to delimit requests
    lttng enable-channel --userspace my-usr-channel
    lttng enable-event --userspace httpd:* --channel=my-usr-channel
    lttng add-context --userspace --channel=my-usr-channel --type=vtid --type=vpid

    # Enable all system calls with their arguments
    lttng enable-channel --kernel --subbuf-size=128M --num-subbuf=8 my-kernel-channel
    lttng enable-event --kernel --channel=my-kernel-channel --syscall --all
    lttng add-context --kernel --channel=my-kernel-channel --type=procname
    lttng add-context --kernel --channel=my-kernel-channel --type=pid
    lttng add-context --kernel --channel=my-kernel-channel --type=tid

    # Start tracing for the specified duration
    read -p "Press enter to start tracing for $duration s"
    lttng start
    sleep $duration
    lttng stop
    lttng destroy my-session
    ```

2. On the web server, execute the installation script:
    ```bash
    bash trace.sh
    ```

3. On the web server, press `Enter` to start tracing

4. On the client, call `wrk` to send requests:
    ```bash
    ./wrk2/wrk -t8 -c10 -d10s -R100 http://@HOST
    ```

5. On the web server, press `Enter` to stop tracing once `wrk` has finished
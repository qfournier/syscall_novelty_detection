# Configuring the webserver to generate Out-Of-Distribuition (OOD) scenarios

Several tools are used to induce the web server to behave anomally. We focused on misconfiguration and performance issues behaviors for now.

## Prerequisites

For the performance OOD, make sure that you have 'stress-ng' installed (see [https://github.com/ColinIanKing/stress-ng](https://github.com/ColinIanKing/stress-ng) for additional information).

1. Install stress-ng:
    ```bash
    sudo apt-get update
    sudo apt-get install stress-ng
    ```

## CPU overload - OOD CPU

1. On the web server, run stress-ng to create start N workers performing various matrix operations on floating point values.

    ```bash
    stress-ng --matrix 48 --matrix-method prod --matrix-size 256 --timeout 120
    ```

2. On the web server, execute the trace collection script:
    ```bash
    bash trace.sh
    ```

3. On the client, call `wrk` to send requests, e.g.,
    ```bash
    ./wrk2/wrk -t8 -c10 -d10s -R100 http://@HOST
    ```

4. On the web server, press `Enter` to start tracing

## Disabling OPCache on PHP - OOD OPCache

### Enable the OOD

1. Remove from `/opt/php-8.0.5/lib/php.ini` file:
    ```
    zend_extension=/opt/php-8.0.5/lib/php/extensions/no-debug-non-zts-20200930/opcache.so
    opcache.enable = 1
    opcache.enable_cli = 1
    opcache.memory_consumption = 128
    opcache.interned_strings_buffer = 8
    opcache.max_accelerated_files = 10000
    opcache.use_cwd = 0
    opcache.validate_timestamps = 0
    opcache.save_comments = 0
    opcache.load_comments = 0
    opcache.enable_file_override = 1
    ```

2. Stop and start `PHP`:
    ```bash
    sudo kill `pgrep php`
    sudo /opt/php-8.0.5/sbin/php-fpm --fpm-config /opt/php-8.0.5/etc/php-fpm.conf
    ```

### Disable the OOD

1. In the `/opt/php-8.0.5/lib/php.ini` file, add the following lines:
    ```
    zend_extension=/opt/php-8.0.5/lib/php/extensions/no-debug-non-zts-20200930/opcache.so
    opcache.enable = 1
    opcache.enable_cli = 1
    opcache.memory_consumption = 128
    opcache.interned_strings_buffer = 8
    opcache.max_accelerated_files = 10000
    opcache.use_cwd = 0
    opcache.validate_timestamps = 0
    opcache.save_comments = 0
    opcache.load_comments = 0
    opcache.enable_file_override = 1
    ```

2. Stop and start `PHP`:
    ```bash
    sudo kill `pgrep php`
    sudo /opt/php-8.0.5/sbin/php-fpm --fpm-config /opt/php-8.0.5/etc/php-fpm.conf
    ```

## Dump I/O misconfiguration - OOD IO

### Enable the OOD

1. Change `httpd`'s config file:
    ```bash
    sudo sed -i '96s#.*#LoadModule dumpio_module modules/mod_dumpio.so#' /opt/httpd/conf/httpd.conf
    ```

2. In the `/opt/httpd/conf/httpd.conf` file, add the lines:
    ```
    <IfModule dumpio_module>
        LogLevel dumpio:trace7
        DumpIOInput On
        DumpIOOutput On
    </IfModule>
    ```

3. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

5. Look if "DumpIO" is on by looking the httpd logs when accessing any page.

### Disable the OOD

1. Change `httpd`'s config file:
    ```bash
    sudo sed -i '96s#.*#\#LoadModule dumpio_module modules/mod_dumpio.so#' /opt/httpd/conf/httpd.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```


## Limiting the number of connections - OOD Connection

### Enable the OOD

1. Change `httpd`'s config file:
    ```bash
    sudo sed -i '463s#.*#Include conf/extra/httpd-mpm.conf#' /opt/httpd/conf/httpd.conf
    sudo sed -i '66s#.*#    MaxRequestWorkers       25#' /opt/httpd/conf/extra/httpd-mpm.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

### Disable the OOD

1. Change `httpd`'s config files:
    ```bash
    sudo sed -i '66s#.*#    MaxRequestWorkers      400#' /opt/httpd/conf/extra/httpd-mpm.conf
    sudo sed -i '463s#.*#\#Include conf/extra/httpd-mpm.conf#' /opt/httpd/conf/httpd.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

## Creating a new socket for each connection - OOD Socket

### Enable the OOD

1. Change `httpd`'s config files:
    ```bash
    sudo sed -i '490s#.*#Include conf/extra/httpd-default.conf#' /opt/httpd/conf/httpd.conf
    sudo sed -i '16s#.*#KeepAlive Off#' /opt/httpd/conf/extra/httpd-default.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

### Disable the OOD

1. Change `httpd`'s config files:
    ```bash
    sudo sed -i '490s#.*#\#Include conf/extra/httpd-default.conf#' /opt/httpd/conf/httpd.conf
    sudo sed -i '16s#.*#KeepAlive On#' /opt/httpd/conf/extra/httpd-default.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

## Disabling SSL in httpd - OOD SSL

### Enable the OOD

1. Change `httpd`'s config file:

    ```bash
    sudo sed -i '137s#.*#\#LoadModule ssl_module modules/mod_ssl.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '498s#.*#\#Include conf/extra/httpd-ssl.conf#' /opt/httpd/conf/httpd.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

### Disable the OOD

1. Change `httpd`'s config file:

    ```bash
    sudo sed -i '137s#.*#LoadModule ssl_module modules/mod_ssl.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '498s#.*#Include conf/extra/httpd-ssl.conf#' /opt/httpd/conf/httpd.conf
    ```

2. Stop and start `httpd`:
    ```bash
    sudo kill `pgrep httpd`
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```
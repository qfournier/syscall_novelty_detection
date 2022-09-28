# Deploy the Web Server to Generate Data

For additional information, see [https://ivopetkov.com/b/install-php-and-apache-from-source/](https://ivopetkov.com/b/install-php-and-apache-from-source/)

## Prerequisites

This script assumes a fresh install Ubuntu Server 21.04 with a kernel 5.11.0

1. Update the server:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade -y
    ```

2. Update the kernel:
    ```bash
    mkdir linux-5.12.5
    cd linux-5.12.5
    wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.12.5/amd64/linux-headers-5.12.5-051205-generic_5.12.5-051205.202105190541_amd64.deb https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.12.5/amd64/linux-headers-5.12.5-051205_5.12.5-051205.202105190541_all.deb https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.12.5/amd64/linux-image-unsigned-5.12.5-051205-generic_5.12.5-051205.202105190541_amd64.deb https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.12.5/amd64/linux-modules-5.12.5-051205-generic_5.12.5-051205.202105190541_amd64.deb
    sudo dpkg -i *.deb
    ```

3. Reboot and check if the kernel has been updated to 5.12.5:
    ```bash
    sudo reboot
    uname -r
    ```

2. Install the dependencies:
    ```bash
    sudo apt-get install -y gcc make libapr1 libapr1-dev libaprutil1 libaprutil1-dev libpcre3 libpcre3-dev libxml2 libxml2-dev pkg-config sqlite3 libsqlite3-dev zlib1g zlib1g-dev libssl-dev openssl 
    ```

3. Create a user that will be used by [Apache](https://httpd.apache.org) and [PHP](https://www.php.net) and the directory where our files will be located:
    ```bash
    sudo mkdir /var/www
    sudo groupadd www-group
    sudo useradd -d /var/www -g www-group -s /bin/false www-user
    sudo chown www-user:www-group /var/www
    ```

4. Clean up:
    ```bash
    cd /home/$USER
    rm -r /home/$USER/linux-5.12.5/
    ```

## LTTng

1. Add the repository for [LTTng](https://lttng.org):
    ```bash
    sudo apt-add-repository -y ppa:lttng/stable-2.12
    ```

2. Install LTTng:
    ```bash
    git clone https://github.com/qfournier/lttng-modules.git
    cd lttng-modules/
    git checkout headers 
    make -j 8
    sudo make modules_install
    sudo depmod -a
    sudo apt-get install -y lttng-tools liblttng-ust-dev
    ```


3. Create a group `tracing` if it doesn't exist and add the current user to the group
    ```bash
    sudo groupadd -r tracing
    sudo usermod -aG tracing $USER
    ```

4. Clean up:
    ```bash
    cd /home/$USER
    rm -rf /home/$USER/lttng-modules/
    ```

## PHP

1. Create the directory where [PHP](https://www.php.net) will be installed:
    ```bash
    sudo mkdir /opt/php-8.0.5
    ```

2. Download the [PHP](https://www.php.net) source and extract it to the target directory:
    ```bash
    wget http://us.php.net/get/php-8.0.5.tar.bz2/from/this/mirror -O /home/$USER/php-8.0.5.tar.bz2
    tar -jxf /home/$USER/php-8.0.5.tar.bz2  -C /home/$USER
    ```

3. Change the current directory:
    ```bash
    cd /home/$USER/php-8.0.5
    ```

4. Configure, build and install:
    ```bash
    ./configure --prefix=/opt/php-8.0.5 --enable-fpm --enable-opcache --enable-mysqlnd --with-mysqli --with-pdo-mysql --with-mysql-sock="/var/run/mysqld/mysqld.sock" --without-pdo-sqlite
    make -j 8
    sudo make install
    ```

5. Create the configuration files (or copy the recommended ones) to their proper places:
    ```bash
    sudo cp /home/$USER/php-8.0.5/php.ini-production /opt/php-8.0.5/lib/php.ini
    sudo cp /opt/php-8.0.5/etc/php-fpm.conf.default /opt/php-8.0.5/etc/php-fpm.conf
    ```

6. In the `/opt/php-8.0.5/lib/php.ini` file, add the following lines IF YOU WANT TO ENABLE OPCACHE:
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

7. Remove the last line of `/opt/php-8.0.5/etc/php-fpm.conf` which should be `include=/opt/php-8.0.5/etc/php-fpm.d/*.conf`:
    ```bash
    sudo sh -c "head -n -1 /opt/php-8.0.5/etc/php-fpm.conf > /opt/php-8.0.5/etc/php-fpm-tmp.conf"
    sudo mv /opt/php-8.0.5/etc/php-fpm-tmp.conf /opt/php-8.0.5/etc/php-fpm.conf
    ```

8. Add the following lines to `/opt/php-8.0.5/etc/php-fpm.conf`:
    ```
    pid = run/php-fpm.pid
    [www]
    user = www-user
    group = www-group
    listen = 127.0.0.1:8999
    pm = dynamic
    pm.max_children = 10
    pm.start_servers = 2
    pm.min_spare_servers = 2
    pm.max_spare_servers = 4
    ```

9. Start PHP-FPM:
    ```bash
    sudo /opt/php-8.0.5/sbin/php-fpm --fpm-config /opt/php-8.0.5/etc/php-fpm.conf
    ```

10. Clean up:
    ```bash
    cd /home/$USER
    rm -r /home/$USER/php-8.0.5.tar.bz2 /home/$USER/php-8.0.5/
    ```

## Apache

1. Copy `httpd.tar.gz` from the `resources` folder into the home directory of the web server.

2. Create the directory where [Apache](https://httpd.apache.org) will be installed:
    ```bash
    sudo mkdir /opt/httpd
    ```

3. Extract `httpd` to the target directory:
    ```bash
    tar -xvf /home/$USER/httpd.tar.gz -C /home/$USER
    ```

4. Download some dependencies:
    ```bash
    wget https://archive.apache.org/dist/apr/apr-1.5.2.tar.bz2 -O /home/$USER/httpd/srclib/apr-1.5.2.tar.bz2
    tar -jxf /home/$USER/httpd/srclib/apr-1.5.2.tar.bz2 -C /home/$USER/httpd/srclib
    mv /home/$USER/httpd/srclib/apr-1.5.2 /home/$USER/httpd/srclib/apr
    wget https://archive.apache.org/dist/apr/apr-util-1.5.4.tar.bz2 -O /home/$USER/httpd/srclib/apr-util-1.5.4.tar.bz2
    tar -jxf /home/$USER/httpd/srclib/apr-util-1.5.4.tar.bz2 -C /home/$USER/httpd/srclib
    mv /home/$USER/httpd/srclib/apr-util-1.5.4 /home/$USER/httpd/srclib/apr-util
    ```
    
5. Change the current directory:
    ```bash
    cd /home/$USER/httpd/
    ```

6. Generate the SSL certificates and set the COMMON_NAME to be the name of the machine (e.g., compute3.dorsal.polymtl.ca):
    ```bash
    sudo mkdir /opt/httpd/certificate
    cd /opt/httpd/certificate 
    sudo openssl req -new -newkey rsa:4096 -x509 -sha256 -days 365 -nodes -out ca.crt -keyout ca.key
    ```

7. Configure `httpd` and edit the Makefile to include the tracepoints dependencies:
    ```bash
    cd /home/$USER/httpd/
    ./configure --prefix=/opt/httpd --enable-so --enable-ssl -enable-dumpio --enable-ratelimit 
    sed -i '12s#.*#PROGRAM_LDADD        = buildmark.o $(HTTPD_LDFLAGS) $(PROGRAM_DEPENDENCIES) $(PCRE_LIBS) -llttng-ust -ldl $(EXTRA_LIBS) $(AP_LIBS) $(LIBS)#' /home/$USER/httpd/Makefile
    sed -i '18s#.*#  os/$(OS_DIR)/libos.la \\#' /home/$USER/httpd/Makefile
    sed -i '19s#.*#  tracepoint/httpdtp.a#' /home/$USER/httpd/Makefile
    ```

8. Build the tracepoints:
    ```bash
    cd /home/$USER/httpd/tracepoint
    make -j 8
    cd /home/$USER/httpd/
    ```

9. Build and install `httpd`:
    ```bash
    make -j 8
    sudo make install
    ```

10. Verify that the tracepoints are available:
    ```bash
    nm /opt/httpd/bin/httpd | grep tracepoint_httpd
    ```

11. Configure `httpd`:
    ```bash
    sudo sed -i '89s#.*#LoadModule socache_shmcb_module modules/mod_socache_shmcb.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '119s#.*#LoadModule proxy_module modules/mod_proxy.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '123s#.*#LoadModule proxy_fcgi_module modules/mod_proxy_fcgi.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '137s#.*#LoadModule ssl_module modules/mod_ssl.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '156s#.*#LoadModule rewrite_module modules/mod_rewrite.so#' /opt/httpd/conf/httpd.conf
    sudo sed -i '166s#.*#User www-user#' /opt/httpd/conf/httpd.conf
    sudo sed -i '167s#.*#Group www-group#' /opt/httpd/conf/httpd.conf
    sudo sed -i '222s#.*#DocumentRoot "/var/www"#' /opt/httpd/conf/httpd.conf
    sudo sed -i '223s#.*#<Directory "/var/www">#' /opt/httpd/conf/httpd.conf
    sudo sed -i '498s#.*#Include conf/extra/httpd-ssl.conf#' /opt/httpd/conf/httpd.conf
    sudo sh -c "echo 'ProxyPassMatch ^/(.*\.php(/.*)?)$ fcgi://127.0.0.1:8999/var/www/\$1' >> /opt/httpd/conf/httpd.conf"
    sudo sed -i '52s#.*#SSLCipherSuite RC4-SHA:AES128-SHA:HIGH:!aNULL:!MD5#' /opt/httpd/conf/extra/httpd-ssl.conf
    sudo sed -i '144s#.*#SSLCertificateFile "/opt/httpd/certificate/ca.crt"#' /opt/httpd/conf/extra/httpd-ssl.conf
    sudo sed -i '154s#.*#SSLCertificateKeyFile "/opt/httpd/certificate/ca.key"#' /opt/httpd/conf/extra/httpd-ssl.conf
    ```

15. Start [Apache](https://httpd.apache.org) (see [here](https://lttng.org/docs/v2.12/#doc-using-lttng-ust-with-daemons)):
    ```bash
    sudo LD_PRELOAD=liblttng-ust-fork.so /opt/httpd/bin/httpd
    ```

16. Clean up:
    ```bash
    cd /home/$USER
    rm -r /home/$USER/httpd.tar.gz /home/$USER/httpd/
    ```

## MySQL

1. Install [MySQL](https://www.mysql.com/fr/):
    ```bash
    sudo apt-get install -y mysql-server
    sudo mysql_secure_installation --use-default --password='password'
    ```

2. Start [MySQL](https://www.mysql.com/fr/) service and launch at reboot:
    ```bash
    sudo systemctl start mysql
    sudo systemctl enable mysql
    ```

3. Disable password verification:
    ```bash
    sudo mysql --user='root' --password='password' --execute "UNINSTALL COMPONENT 'file://component_validate_password';"
    ```

4. Create a user:
    ```bash
    sudo mysql --user='root' --password='password' --execute "CREATE USER 'user'@'localhost' IDENTIFIED WITH mysql_native_password BY 'password';"
    sudo mysql --user='root' --password='password' --execute "GRANT ALL PRIVILEGES ON * . * TO 'user'@'localhost';"
    ```

5. Fill [MySQL](https://www.mysql.com/fr/) with random data:
    ```bash
    wget https://downloads.mysql.com/docs/sakila-db.tar.gz
    tar -xzf sakila-db.tar.gz
    mysql --user='user' --password='password' < sakila-db/sakila-schema.sql
    mysql --user='user' --password='password' < sakila-db/sakila-data.sql
    ```

6. Clean up:
    ```bash
    cd /home/$USER
    rm -r /home/$USER/sakila-db.tar.gz /home/$USER/sakila-db 
    ```

## Web Page

Add the following lines to `/var/www/index.php`.

```html
<head></head>
<body>
Hello, the random actor is:<br>
<?php
    $id = rand(1, 200);
    $db = new PDO("mysql:host=localhost;dbname=sakila;charset=utf8", "user", "password");
    $author = $db->query("SELECT first_name, last_name FROM actor where actor_id=$id");
    while($row = $author->fetch(PDO::FETCH_ASSOC)) {
        echo "First name: {$row["first_name"]}  <br>";
        echo "Last name: {$row["last_name"]}  <br>";
    }
?>
</body>
</html>
```

## Novel Behvaiours

To generate novel behaviours, see the [setup-ood.md](./setup-ood.md)
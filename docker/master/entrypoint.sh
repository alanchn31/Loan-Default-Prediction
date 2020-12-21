#!/bin/bash

# Configure Livy based on environment variables
if [[ -n "${SPARK_MASTER}" ]]; then
  echo "livy.spark.master=${SPARK_MASTER}" >> "${LIVY_CONF_DIR}/livy.conf"
fi
if [[ -n "${SPARK_DEPLOY_MODE}" ]]; then
  echo "livy.spark.deploy-mode=${SPARK_DEPLOY_MODE}" >> "${LIVY_CONF_DIR}/livy.conf"
fi
if [[ -n "${LOCAL_DIR_WHITELIST}" ]]; then
  echo "livy.file.local-dir-whitelist=${LOCAL_DIR_WHITELIST}" >> "${LIVY_CONF_DIR}/livy.conf"
fi
if [[ -n "${ENABLE_HIVE_CONTEXT}" ]]; then
  echo "livy.repl.enable-hive-context=${ENABLE_HIVE_CONTEXT}" >> "${LIVY_CONF_DIR}/livy.conf"
fi
if [[ -n "${LIVY_HOST}" ]]; then
  echo "livy.server.host=${LIVY_HOST}" >> "${LIVY_CONF_DIR}/livy.conf"
fi
if [[ -n "${LIVY_PORT}" ]]; then
  echo "livy.server.port=${LIVY_PORT}" >> "${LIVY_CONF_DIR}/livy.conf"
fi

"$LIVY_HOME/bin/livy-server" $@
set -e

for f in *.mcap; do
  record_name="${f%.mcap}"
  record_id=$(cocli record create --title $record_name | grep "Record created:" | awk '{print $3}')
  cocli record upload $record_id $f #上传文件
  cocli record update $record_id -l mcap #打标签
done

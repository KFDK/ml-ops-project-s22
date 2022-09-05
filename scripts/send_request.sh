curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{ "instances": [ { "data": { "string": "Take away the CGI and the A-list cast and you end up with a film with less punch." } } ] }' \
  http://localhost:8080/predictions/model/

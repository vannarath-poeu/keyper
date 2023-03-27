Part 1: Starting Lucence
1. Run `cd docker`.
2. Run `docker-compose up` to start lucene in docker.
(Optional)
- For creating core, run `docker exec -it --user solr keyper-lucene bin/solr create_core -c $CORE_NAME`
- For deleting core, run `docker exec -it --user solr keyper-lucene bin/solr delete -c $CORE_NAME`

Part 2: Running scripts
1. Run `cd src`.

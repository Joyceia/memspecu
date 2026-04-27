def recalculate_metrics(self):
    # Keep existing functionality for legacy metrics
    legacy_metrics = self.get_action_metrics(k=None)

    # Compute overall actions_top1 and actions_top3
    actions_top1 = self.get_action_metrics(k=1)['general']
    actions_top3 = self.get_action_metrics(k=3)['general']

    # Prepare the metrics data to be written to metrics.json
    metrics_data = {
        "legacy_metrics": legacy_metrics,
        "actions_top1": actions_top1,
        "actions_top3": actions_top3
    }

    # Write to metrics.json
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
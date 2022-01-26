(define (domain sorting_domain)
    (:requirements :equality)
    (:predicates (CanAtTarget ?can - Can ?target - Target)
                 (CanInGripper ?can - Can)
                 (CanObstructs ?can1 - Can ?can2 - Can)
                 (CanObstructsTarget ?can - Can ?target - Target)
                 (CanInReach ?can - Can)
                 (NearCan ?can - Can)
                 (WaitingOnCan ?waiter - Can ?obstr - Can)
                 (WaitingOnTarget ?waiter - Can ?obstr - Target)
    )

    (:types Can Target)

    (:action grasp
        :parameters (?can - Can)
        :precondition (forall (?c - Can) (not (CanObstructs ?c ?can)))
        :effect (and (CanInGripper ?can)
                     (forall (?c - Can) (when (not (= ?c ?can)) (not (CanInGripper ?c))))
                     (NearCan ?can)
                     (forall (?c - Can) (when (not (= ?c ?can)) (not (NearCan ?c)))))
    )

    (:action putdown
        :parameters (?can - Can ?target - Target)
        :precondition (and (NearCan ?can)
                           (forall (?c - Can) (not (CanAtTarget ?c ?target)))
                           (forall (?c - Can) (not (CanObstructsTarget ?c ?target)))
                           (forall (?c - Can) (not (WaitingOnCan ?can ?c)))
                           (forall (?t - Target) (not (WaitingOnTarget ?can ?t))))
        :effect (and (not (CanInGripper ?can))
                     (forall (?c - Can) (when (CanObstructs ?can ?c) (WaitingOnCan ?can ?c)))
                     (forall (?t - Target) (when (CanObstructsTarget ?can ?t) (WaitingOnTarget ?can ?t)))
                     (forall (?c - Can) (not (WaitingOnCan ?c ?can)))
                     (forall (?c - Can) (not (WaitingOnTarget ?c ?target)))
                     (CanAtTarget ?can ?target)
                     (forall (?c - Can) (not (CanObstructs ?can ?c)))
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (CanAtTarget ?can ?t))))
                     (forall (?t - Target) (when (not (= ?t ?target)) (not (CanObstructsTarget ?can ?t))))
                     (forall (?c - Can) (when (not (= ?c ?can)) (not (NearCan ?c)))))
    )
)
